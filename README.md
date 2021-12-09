# automl
AutoML is a web-servie that impliments online machine learning pipeline. It trains several models and selects one based on user's metric.

At the moment automl service only works with binary classification problem and supports metrics **accuracy**, **precision**, **recall** and **f1**
This model does not use feature selection and it does not fill missing data.

# Installation
## Regular Installation
This service requires python3.8 or higher.

create virtual enviroment:
```
python3 -m venv venv
```

activate virtual enviroment

On Linux:
```
source venv/bin/activate
```

On Windows:
```
venv\Script\activate
```

install required libraries:
```
python3 -m pip install -r requirements.txt
```

go to automl dir:
```
cd automl
```

and start an app:
```
python3 -m flask run
```

## Docker

You can set up this web-service in docker container if you wish.

For this you will need [docker-compose](https://docs.docker.com/compose/install/)

Then run 
```
docker-compose up --build
```

And your service should work on http://localhost:5000

# Usage
You can find usage example in the example.ipynb 

Make sure to run web-server before you send any request to it
