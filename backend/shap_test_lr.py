import shap
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt

# Load the dataset
file_path_train = "mitbih_train.csv"
file_path_test = "mitbih_test.csv"
df_train = pd.read_csv(file_path_train)
df_train = df_train.dropna()
df_test = pd.read_csv(file_path_test)
df_test = df_test.dropna()

# Prepare data
X_train = df_train.iloc[:, :-1].values  # 187 time points
y_train = df_train.iloc[:, -1].values
X_test = df_test.iloc[:, :-1].values
y_test = df_test.iloc[:, -1].values

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)  # Use transform for test data

class_names = ["Normal","Supravitricular","Vitricular","Fusion","Unknow"]

# Load the model
clf = joblib.load('LogisticRegression_classweight.pkl')

# Select a representative background dataset (100 samples)
background = X_train[np.random.choice(X_train.shape[0], 100, replace=False)]

# Create a SHAP LinearExplainer with an Independent masker
masker = shap.maskers.Independent(data=background)
explainer = shap.LinearExplainer(clf, masker=masker)

print("Explainer created")
# Select a single test instance for visualization (e.g., first instance)
x_instance = X_test[0:1]  # Shape: (1, 187)
shap_values = explainer(x_instance)  # shap_values is an Explanation object

# For multi-class, shap_values is a list of Explanation objects (one per class)
# Select the class with the highest predicted probability for visualization
pred_probs = clf.predict_proba(x_instance)[0]
class_idx = np.argmax(pred_probs)  # Class with highest probability

# Extract SHAP values and input data for the selected instance and class
shap_vals = shap_values[0, :, class_idx]  # SHAP values for first instance, selected class
ecg_signal = x_instance[0]  # Scaled ECG signal

# Create time points (0 to 186) for the x-axis
time_points = np.arange(X_train.shape[1])

# Plot SHAP values and ECG signal
plt.figure(figsize=(12, 6))

# Plot the ECG signal
plt.plot(time_points, ecg_signal, label="Scaled ECG Signal", color="blue", alpha=0.5)

# Plot SHAP values as a line plot
plt.plot(time_points, shap_vals.values, label=f"SHAP Values (Class {class_names[int(y_test[0])]})", color="red")

# Add labels and title
plt.xlabel("Time Point")
plt.ylabel("Value")
plt.title(f"SHAP Values for ECG Time-Series (Class {class_names[int(y_test[0])]})")
plt.legend()
plt.grid(True)
plt.show()