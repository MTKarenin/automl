import pickle
import pandas as pd
from .models import ClassificationInputModel
from sklearn import model_selection, ensemble, metrics


class NotSupportedEstimatorException(Exception):
    pass


class Classifier:
    def __init__(self, model, score, score_type) -> None:
        self.model = model
        self.score = score
        self.score_type = score_type

    def __repr__(self) -> str:
        return f'{str(self.model)} score: {self.score_type} = {self.score}'

    def _to_pickle_string(self) -> str:
        return pickle.dumps(self.model)



class ClassifierFactory:
    estimators = (
        ensemble.GradientBoostingClassifier,
        ensemble.RandomForestClassifier,
    )

    hyperparameters = {
        ensemble.GradientBoostingClassifier: {
            'learning_rate':[0.15,0.1,0.05,0.01,0.005,0.001],
            'n_estimators':[100,250,500,750,1000,1250,1500,1750]
        },
        ensemble.RandomForestClassifier: {
            'max_depth': [2, 3, 4, 5, 6, 7, 8],
            'min_samples_split': [2, 3, 4, 5, 6, 7, 8]
        }
    }

    scores = {
        'accuracy': metrics.accuracy_score, 
        'precision': metrics.precision_score, 
        'recall': metrics.recall_score, 
        'f1': metrics.f1_score
    }

    def __init__(self) -> None:
        pass

    def make_classifier(self, params: ClassificationInputModel, classifier_type) -> Classifier:
        self._check_classifier_type_in_supported_estimators(classifier_type)

        model = classifier_type()
        data = pd.read_json(params.data)
        data_train, data_test = model_selection.train_test_split(data)
        X, y = data_train[params.features], data_train[params.target]

        random_search = model_selection.RandomizedSearchCV(
                model, 
                param_distributions=self.hyperparameters[classifier_type], 
                n_iter=20, 
                scoring=params.metric
            ).fit(X, y)

        model = random_search.best_estimator_
        score = self.scores[params.metric](data_test[params.target], model.predict(data_test[params.features]))

        model = classifier_type(**random_search.best_params_)
        model.fit(data[params.features], data[params.target])

        return Classifier(model, score, params.metric)

    @classmethod
    def _check_classifier_type_in_supported_estimators(cls, classifier_type: str):
        if classifier_type not in cls.estimators:
            raise NotSupportedEstimatorException(f'supported estimators are {cls.estimators}')


def do_autoclassification(params, model_id):
    try:
        classifier_factory = ClassifierFactory()
        classifiers = []
        for classifier_type in classifier_factory.estimators:
            classifier = classifier_factory.make_classifier(params, classifier_type)
            classifiers.append(classifier)
            print(classifier)

        best_score = max([model.score for model in classifiers])
        best_model = [model for model in classifiers if model.score == best_score][0]

        with open(f'saved_models/{model_id}.pickle', 'wb') as file:
            pickle.dump(best_model, file)
        print(f'model {model_id} calculated')
    except Exception as e:
        with open(f'error_logs/{model_id}.log', 'w') as file:
            file.write(str(e))
        raise e
