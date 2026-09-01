"""Unit tests for Convolutional Neural Network. Minimum 80% coverage."""

import numpy as np
import pytest

from models.group_07_cnn.model import CNNModel


@pytest.fixture
def training_data():
    """Small synthetic 28x28 grayscale dataset."""
    rng = np.random.default_rng(42)

    X = rng.random((12, 784), dtype=np.float32)
    y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])

    return X, y


@pytest.fixture
def fitted_model(training_data):
    """Return a trained CNN model."""
    X, y = training_data

    model = CNNModel(epochs=1, learning_rate=0.001, random_state=42)
    model.fit(X, y)

    return model


def test_model_initialization():
    """Model should initialize with the expected defaults."""
    model = CNNModel()

    assert model.epochs == 10
    assert model.learning_rate == 0.001
    assert model.random_state == 42
    assert model.is_fitted is False


def test_invalid_epochs():
    """Epochs must be positive."""
    with pytest.raises(ValueError, match="epochs"):
        CNNModel(epochs=0)

    with pytest.raises(ValueError, match="epochs"):
        CNNModel(epochs=-1)


def test_invalid_learning_rate():
    """Learning rate must be positive."""
    with pytest.raises(ValueError, match="learning_rate"):
        CNNModel(learning_rate=0)

    with pytest.raises(ValueError, match="learning_rate"):
        CNNModel(learning_rate=-0.001)


def test_invalid_X_type():
    """X must be a NumPy array."""
    model = CNNModel()

    with pytest.raises(TypeError, match="numpy ndarray"):
        model.fit([[1, 2, 3]], np.array([0]))


def test_invalid_X_dimensions():
    """X must be a two-dimensional array."""
    model = CNNModel()

    X = np.zeros((2, 1, 784))

    with pytest.raises(ValueError, match="2D"):
        model.fit(X, np.array([0, 1]))


def test_empty_X():
    """X must contain at least one sample."""
    model = CNNModel()

    X = np.empty((0, 784))

    with pytest.raises(ValueError, match="at least one sample"):
        model.fit(X, np.array([], dtype=int))


def test_wrong_feature_count():
    """CNN expects flattened 28x28 images with 784 features."""
    model = CNNModel()

    X = np.zeros((2, 783))

    with pytest.raises(ValueError, match="784"):
        model.fit(X, np.array([0, 1]))


def test_non_finite_X():
    """X must not contain NaN or infinite values."""
    model = CNNModel()

    X = np.zeros((2, 784))
    X[0, 0] = np.nan

    with pytest.raises(ValueError, match="non-finite"):
        model.fit(X, np.array([0, 1]))


def test_fit_requires_y(training_data):
    """CNN is supervised and therefore requires labels."""
    X, _ = training_data
    model = CNNModel(epochs=1)

    with pytest.raises(ValueError, match="y must not be None"):
        model.fit(X)


def test_invalid_y(training_data):
    """Validate label type, shape and length."""
    X, y = training_data
    model = CNNModel(epochs=1)

    with pytest.raises(TypeError, match="numpy ndarray"):
        model.fit(X, list(y))

    with pytest.raises(ValueError, match="1D"):
        model.fit(X, y.reshape(-1, 1))

    with pytest.raises(ValueError, match="rows"):
        model.fit(X, y[:-1])

    with pytest.raises(ValueError, match="non-finite"):
        bad_y = y.astype(float)
        bad_y[0] = np.nan
        model.fit(X, bad_y)


def test_single_class_rejected(training_data):
    """Classification requires at least two classes."""
    X, _ = training_data
    y = np.zeros(X.shape[0], dtype=int)

    model = CNNModel(epochs=1)

    with pytest.raises(ValueError, match="at least two classes"):
        model.fit(X, y)


def test_predict_before_fit():
    """Prediction before training should fail."""
    model = CNNModel()

    X = np.zeros((2, 784))

    with pytest.raises(RuntimeError, match="fit"):
        model.predict(X)


def test_predict_proba_before_fit():
    """Probability prediction before training should fail."""
    model = CNNModel()

    X = np.zeros((2, 784))

    with pytest.raises(RuntimeError, match="fit"):
        model.predict_proba(X)


def test_fit(training_data):
    """Model should train successfully."""
    X, y = training_data

    model = CNNModel(epochs=1)
    result = model.fit(X, y)

    assert result is model
    assert model.is_fitted is True
    assert model.n_features == 784
    assert model.classes_ is not None
    assert model._train_time is not None
    assert model._train_time >= 0


def test_predict(fitted_model, training_data):
    """Predict should return class labels."""
    X, y = training_data

    predictions = fitted_model.predict(X)

    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (X.shape[0],)
    assert np.all(np.isin(predictions, np.unique(y)))


def test_predict_proba(fitted_model, training_data):
    """Predict_proba should return valid class probabilities."""
    X, y = training_data

    probabilities = fitted_model.predict_proba(X)

    assert isinstance(probabilities, np.ndarray)
    assert probabilities.shape == (X.shape[0], len(np.unique(y)))
    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)
    np.testing.assert_allclose(
        probabilities.sum(axis=1),
        np.ones(X.shape[0]),
        atol=1e-6,
    )


def test_metadata(fitted_model):
    """Metadata should contain the required platform fields."""
    metadata = fitted_model.get_metadata()

    assert metadata["model_name"] == "Convolutional Neural Network"
    assert metadata["model_type"] == "classifier"
    assert metadata["n_features"] == 784
    assert metadata["training_time_seconds"] is not None
    assert metadata["feature_importance"] is None
    assert metadata["hyperparameters"] is not None


def test_reproducible_training(training_data):
    """Same random state and data should produce identical predictions."""
    X, y = training_data

    model1 = CNNModel(epochs=1, random_state=42)
    model2 = CNNModel(epochs=1, random_state=42)

    model1.fit(X, y)
    model2.fit(X, y)

    predictions1 = model1.predict(X)
    predictions2 = model2.predict(X)

    np.testing.assert_array_equal(predictions1, predictions2)