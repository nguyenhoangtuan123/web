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

class_names = ["Normal", "Supraventricular", "Ventricular", "Fusion", "Unknown"]

# Load the model
clf = joblib.load('SVC_classweight.pkl')

# Select a representative background dataset (100 samples)
background = X_train[np.random.choice(X_train.shape[0], 100, replace=False)]

# Create a SHAP KernelExplainer using decision_function
def decision_function(X):
    return clf.decision_function(X)

explainer = shap.KernelExplainer(decision_function, background)

print("Explainer created")

# Select a single test instance for visualization (e.g., first instance)
x_instance = X_test[0:1]  # Shape: (1, 187)
shap_values = explainer.shap_values(x_instance)  # shap_values is a list for multi-class

# For multi-class, shap_values is a list of arrays (one per class)
# Select the class with the highest decision score for visualization
decision_scores = clf.decision_function(x_instance)[0]
class_idx = np.argmax(decision_scores)  # Class with highest decision score

# Extract SHAP values and input data for the selected instance and class
shap_vals = shap_values[class_idx][0]  # SHAP values for first instance, selected class
ecg_signal = x_instance[0]  # Scaled ECG signal

# Create time points (0 to 186) for the x-axis
time_points = np.arange(X_train.shape[1])

# Plot SHAP values and ECG signal
plt.figure(figsize=(12, 6))

# Plot the ECG signal
plt.plot(time_points, ecg_signal, label="Scaled ECG Signal", color="blue", alpha=0.5)

# Plot SHAP values as a line plot
plt.plot(time_points, shap_vals, label=f"SHAP Values (Class {class_names[int(y_test[0])]})", color="red")

# Add labels and title
plt.xlabel("Time Point")
plt.ylabel("Value")
plt.title(f"SHAP Values for ECG Time-Series (Class {class_names[int(y_test[0])]})")
plt.legend()
plt.grid(True)
plt.show()