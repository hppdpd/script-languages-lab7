from abc import ABC, abstractmethod

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class BaseModel(ABC):
    """Контракт (потрібен для LSP).

    fit(x, y) повертає self; predict(x) повертає масив міток
    тієї ж довжини, що й x. Нащадки цей контракт не змінюють.
    """

    name: str

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray) -> "BaseModel": ...

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray: ...


class SklearnModel(BaseModel):
    """Обгортка над будь-яким estimator із scikit-learn."""

    def __init__(self, name: str, estimator: BaseEstimator) -> None:
        self.name = name
        self._est = estimator

    def fit(self, x: np.ndarray, y: np.ndarray) -> "SklearnModel":
        self._est.fit(x, y)  # type: ignore[attr-defined]
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self._est.predict(x))  # type: ignore[attr-defined]


class LogisticModel(SklearnModel):
    def __init__(self) -> None:
        super().__init__(
            "Logistic Regression",
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        )


class ForestModel(SklearnModel):
    def __init__(self) -> None:
        super().__init__(
            "Random Forest",
            RandomForestClassifier(n_estimators=200, random_state=42),
        )
