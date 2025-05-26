import shap
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load sample data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Train Logistic Regression
clf = LogisticRegression(max_iter=200)
clf.fit(X_train, y_train)

# Use KernelExplainer for linear models like Logistic Regression
explainer = shap.Explainer(clf.predict_proba, X_train)
shap_values = explainer(X_test)

# Summary plot
shap.summary_plot(shap_values, X_test)
