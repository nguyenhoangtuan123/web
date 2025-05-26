import shap
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import xgboost as xgb

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

# Load the XGBoost Booster model
clf = joblib.load('XGBoost_classweight.pkl')

# Verify model configuration
try:
    print("Model attributes:", clf.attributes())  # Inspect Booster configuration
    print("Number of features:", clf.num_features())
except AttributeError as e:
    print("Error inspecting model:", e)

# Create a SHAP TreeExplainer
explainer = shap.TreeExplainer(clf)

print("Explainer created")

# Select a single test instance for visualization (e.g., first instance)
x_instance = X_test[20000:20001]  # Shape: (1, 187)
print("x_instance shape:", x_instance.shape)

# Compute SHAP values
shap_values = explainer.shap_values(x_instance)  # shap_values is a list for multi-class

# Inspect shap_values structure
print("shap_values type:", type(shap_values))
if isinstance(shap_values, list):
    print("shap_values length (n_classes):", len(shap_values))
    print("Shape of shap_values[0]:", shap_values[0].shape)
else:
    print("shap_values shape:", shap_values.shape)

# Convert x_instance to DMatrix for Booster's predict method
x_instance_dmatrix = xgb.DMatrix(x_instance)

# Check if the model outputs probabilities (multi:softprob) or class labels (multi:softmax)
try:
    # Try to get probabilities with multi:softprob
    pred_probs = clf.predict(x_instance_dmatrix, output_margin=False)
    print("pred_probs shape:", pred_probs.shape)
    if pred_probs.shape[1] == len(class_names):  # Check if output is probabilities (n_samples, n_classes)
        class_idx = np.argmax(pred_probs[0])  # Class with highest probability
    else:
        raise ValueError("Unexpected output shape from predict")
except:
    # Fallback to predicted class labels (e.g., multi:softmax)
    pred_class = clf.predict(x_instance_dmatrix, output_margin=False)
    print("pred_class shape:", pred_class.shape)
    class_idx = int(pred_class[0])  # Use predicted class

# Extract SHAP values for the selected class
if isinstance(shap_values, list):
    # Multi-class case: shap_values is a list of (n_samples, n_features) arrays
    if len(shap_values) <= class_idx:
        raise ValueError(f"class_idx {class_idx} exceeds shap_values length {len(shap_values)}")
    shap_vals = shap_values[class_idx][0]  # Shape: (187,)
else:
    # Single-class or unexpected case
    shap_vals = shap_values[0]  # Shape: (187,)
    print("Warning: shap_values is not a list, assuming single-class or checking shape")

# Verify shap_vals shape
print("shap_vals shape:", shap_vals.shape)
if shap_vals.shape[0] != X_train.shape[1]:
    raise ValueError(f"SHAP values shape {shap_vals.shape} does not match time_points shape ({X_train.shape[1],})")

ecg_signal = x_instance[0]  # Scaled ECG signal, shape: (187,)

# Create time points (0 to 186) for the x-axis
time_points = np.arange(X_train.shape[1])

# Plot SHAP values and ECG signal
plt.figure(figsize=(12, 6))

# Plot the ECG signal
plt.plot(time_points, ecg_signal, label="Scaled ECG Signal", color="blue", alpha=0.5)

# Plot SHAP values as a line plot
colors = ["red", "green", "orange", "purple", "brown"]
for i, class_name in enumerate(class_names):
	if i == class_idx:
		plt.plot(time_points, shap_vals[:,i], label=f"SHAP Values (Class {class_name})", color=colors[i], linewidth=3)
	else:
		plt.plot(time_points, shap_vals[:,i], label=f"SHAP Values (Class {class_name})", color=colors[i], alpha=0.5)

# Add labels and title
plt.xlabel("Time Point")
plt.ylabel("Value")
plt.title(f"SHAP Values for ECG Time-Series (Class {class_names[class_idx]})")
plt.legend()
plt.grid(True)
plt.show()