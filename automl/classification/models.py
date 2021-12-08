from typing import List
from pydantic import BaseModel, validator


class ClassificationInputModel(BaseModel):
    metric: str
    data: str
    features: List[str]
    target: str

    @validator('metric')
    def metric_in_supported_metrics(cls, v):
        supported_metrics = ('accuracy', 'precision', 'recall', 'f1')
        if v in supported_metrics:
            return v
        else:
            raise KeyError(f'metric {v} is not allowed\nsupported metric are {supported_metrics}')
