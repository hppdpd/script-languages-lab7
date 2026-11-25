import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer

from lab7.detectors import FaceDetector, FrontalAlt, FrontalAlt2, FrontalDefault
from lab7.models import BaseModel, ForestModel, LogisticModel


@pytest.mark.parametrize("model", [LogisticModel(), ForestModel()])
def test_models_follow_contract(model: BaseModel) -> None:
    data = load_breast_cancer()
    assert model.fit(data.data, data.target) is model
    pred = model.predict(data.data[:10])
    assert pred.shape == (10,)
    assert set(np.unique(pred)) <= {0, 1}


@pytest.mark.parametrize("detector", [FrontalDefault(), FrontalAlt2(), FrontalAlt()])
def test_detectors_follow_contract(detector: FaceDetector) -> None:
    blank = np.zeros((200, 200, 3), dtype=np.uint8)
    assert detector.detect(blank) == []
