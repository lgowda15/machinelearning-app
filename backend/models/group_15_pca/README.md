# PCA Dimensionality Reduction (Group 15)

A fast, flexible unsupervised method for dimensionality reduction using Singular Value Decomposition, provided via scikit-learn.

## Usage
```python
from group_15_pca import PCAModel

model = PCAModel(n_components=2)
model.fit(X_train)
X_reduced = model.predict(X_test)
variance = model.get_visualization_data() 




Step 1: Make sure you are inside the backend folder
cd "file location"


Step 2: Run the test using the short module path (no long file path needed)
python -m pytest models/group_15_pca/test.py --cov=models/group_15_pca --cov-report term-missing