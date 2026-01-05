import numpy as np

from src.models.ensemble import InferenceEngine


class DummyModel:
    def predict_proba(self, X):
        # Return a fixed probability vector
        return np.column_stack([1 - 0.7 * np.ones(len(X)), 0.7 * np.ones(len(X))])


def test_inference_engine_returns_probabilities():
    loader = type("Loader", (), {"models": {"xgb": DummyModel(), "rf": DummyModel()}})()
    engine = InferenceEngine(loader, batch_size=2)
    X = np.zeros((4, 3))
    result = engine.predict_batch(X, model_type="ensemble")
    assert result.probabilities.shape == (4,)
    assert np.allclose(result.probabilities, 0.7)
