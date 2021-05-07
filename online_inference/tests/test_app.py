from fastapi.testclient import TestClient
from online_inference.app import app
import pytest


PREDICT_REQUEST_SAMPLE = [
  {
    "id": 0,
    "chol": 0,
    "thalach": 0,
    "oldpeak": 0,
    "trestbps": 0,
    "fbs": 0,
    "age": 0,
    "sex": 0,
    "cp": 0,
    "restecg": 0,
    "exang": 0,
    "slope": 0,
    "ca": 0,
    "thal": 0
  }
]

@pytest.fixture
def client():
    with TestClient(app) as client:
        yield client



def test_health(client) -> None:
    response = client.get("/")
    assert response.status_code == 200
    response = client.get("/unknown")
    assert response.status_code >= 400
    response = client.get("/healz")
    assert response.status_code == 200
    assert response.json() == True
    response = client.get("/predict", json=PREDICT_REQUEST_SAMPLE)
    assert response.status_code == 200
    predict = response.json()[0]
    assert int(predict['id']) == 0
    assert int(predict['target']) == 0 or int(predict['target']) == 1
