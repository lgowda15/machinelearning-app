# Convolutional Neural Network

## Model
A convolutional neural network classifier for image data represented as flattened 28 × 28 grayscale images.

## Usage
```python
from models.group_07_cnn.model import CNNModel
model = CNNModel()
model.fit(X_train, y_train)
preds = model.predict(X_test)
```

## Hyperparameters
| Name            | Default | Controls                                                                                                      |
| --------------- | ------: | ------------------------------------------------------------------------------------------------------------- |
| `epochs`        |    `10` | Number of training passes over the input data. Higher values can improve learning but increase training time. |
| `learning_rate` | `0.001` | Step size used by the Adam optimizer when updating network weights.                                           |
| `random_state`  |    `42` | Seed used to make network initialization and training deterministic.                                          |

# Feature layout
The model expects the standard Group 7 image-model input contract:

X is a 2D NumPy array of shape (n_samples, 784).
Each row contains one flattened 28 × 28 grayscale image.
The 784 features are reshaped internally to (n_samples, 1, 28, 28) before being passed to the convolutional network.
The model performs this reshaping internally; callers should provide the flattened 784-feature representation.
y is a 1D NumPy array containing the class labels.

The feature order is the flattened row-major order of the 28 × 28 image.

## Running the tests
```bash
python -m pytest test.py --cov=. --cov-report=term-missing
```

## Design decisions
A convolutional neural network is used because the input represents images, and convolutional layers can learn local spatial patterns such as edges and shapes.

The network uses two convolutional layers with ReLU activations followed by adaptive average pooling and a linear classifier. Adaptive average pooling reduces the spatial representation to a fixed-size feature vector before classification.

The Adam optimizer is used because it provides adaptive learning-rate updates and works well for training neural networks. The default learning rate is 0.001, and training uses 10 epochs.

The model uses random_state=42 together with PyTorch's deterministic settings so repeated training runs are reproducible.

Changing epochs changes how long the network trains and can affect model performance. Changing learning_rate changes the size of the weight updates and can affect both convergence speed and final performance. Changing random_state changes the initial network weights and therefore may change the learned model while preserving reproducibility for that seed.