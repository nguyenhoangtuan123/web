import shap
import numpy as np
import joblib
import xgboost as xgb
import matplotlib.pyplot as plt

# Preload model and explainer
MODEL_PATH = 'XGBoost_classweight.pkl'
SCALER_PATH = 'scaler.pkl'

# Load XGBoost model
clf = joblib.load(MODEL_PATH)

# Create SHAP TreeExplainer
explainer = shap.TreeExplainer(clf)

# Load pre-fitted scaler
scaler = joblib.load(SCALER_PATH)

def get_shap_values_for_instance(x_instance, model_explainer=explainer, data_scaler=scaler):
    """
    Compute SHAP values for a single test instance from the MIT-BIH dataset using a preloaded XGBoost model and explainer.
    
    Parameters:
    -----------
    x_instance : array-like, shape (1, 187) or (187,)
        Single test instance with 187 time points (unscaled).
    model_explainer : shap.TreeExplainer, optional (default=preloaded explainer)
        Preloaded SHAP TreeExplainer for the XGBoost model.
    data_scaler : sklearn.preprocessing.StandardScaler, optional (default=preloaded scaler)
        Pre-fitted StandardScaler for feature scaling.
    
    Returns:
    --------
    shap_vals_array : np.ndarray, shape (187, 5)
        SHAP values for the instance, where 187 is the number of time points and 5 is the number of classes.
    """
    # Convert input to NumPy array and ensure correct shape
    x_instance = np.array(x_instance)
    if x_instance.shape == (187,):
        x_instance = x_instance.reshape(1, -1)  # Reshape to (1, 187)
    if x_instance.shape != (1, 187):
        raise ValueError(f"Expected x_instance shape (1, 187) or (187,), got {x_instance.shape}")

    # Scale the input instance
    x_instance_scaled = data_scaler.transform(x_instance)  # Shape: (1, 187)
    
    # Compute SHAP values
    shap_values = model_explainer.shap_values(x_instance_scaled)
    
    # Check model configuration
    num_classes = getattr(clf, 'n_classes_', None)
    if num_classes is not None and num_classes != 5:
        raise ValueError(f"Expected 5-class model, got {num_classes} classes")

    x_instance_dmatrix = xgb.DMatrix(x_instance_scaled)
    class_idx = clf.predict(x_instance_dmatrix, output_margin=False)[0]

    # Handle SHAP values based on output format
    if isinstance(shap_values, list):
        # Multi-class, list format (older SHAP versions)
        if len(shap_values) != 5:
            raise ValueError(f"Expected 5 SHAP arrays, got {len(shap_values)} arrays")
        if shap_values[0].shape != (1, 187):
            raise ValueError(f"Expected SHAP shape (1, 187), got {shap_values[0].shape}")
        shap_vals_array = np.stack([shap_values[i][0] for i in range(5)], axis=1)  # Shape: (187, 5)
    elif isinstance(shap_values, np.ndarray) and shap_values.shape == (1, 187, 5):
        # Multi-class, 3D array format (newer SHAP versions)
        shap_vals_array = shap_values[0]  # Extract (187, 5) array
    else:
        raise ValueError(
            f"Unexpected SHAP values format. Got {type(shap_values)} with shape "
            f"{shap_values.shape if hasattr(shap_values, 'shape') else 'N/A'}. "
            f"Expected list of 5 arrays or array with shape (1, 187, 5)."
        )
    
    # Verify output shape
    if shap_vals_array.shape != (187, 5):
        raise ValueError(f"Expected output shape (187, 5), got {shap_vals_array.shape}")
    
    return shap_vals_array, x_instance_scaled, class_idx

def plot_shap_values(x_instance_scaled, shap_vals, clf=clf):
    """
    Plot the scaled ECG signal and SHAP values for each class, highlighting the predicted class.
    
    Parameters:
    -----------
    x_instance_scaled : np.ndarray, shape (1, 187)
        Scaled ECG signal for plotting.
    shap_vals : np.ndarray, shape (187, 5)
        SHAP values for the instance, with 187 time points and 5 classes.
    clf : xgb.XGBClassifier, optional (default=preloaded model)
        XGBoost model to predict the class.
    """
    # Class names for MIT-BIH dataset
    class_names = ["Normal", "Supraventricular", "Ventricular", "Fusion", "Unknown"]
    
    # Predict class for highlighting
    x_instance_dmatrix = xgb.DMatrix(x_instance_scaled)
    pred_probs = clf.predict(x_instance_dmatrix, output_margin=False)
    class_idx = np.argmax(pred_probs[0])  # Predicted class index
    
    # Extract ECG signal and time points
    ecg_signal = x_instance_scaled[0]  # Shape: (187,)
    time_points = np.arange(187)
    
    # Plot
    plt.figure(figsize=(12, 6))
    
    # Plot ECG signal
    plt.plot(time_points, ecg_signal, label="Scaled ECG Signal", color="blue", alpha=0.5)
    
    # Plot SHAP values for each class
    colors = ["red", "green", "orange", "purple", "brown"]
    for i, class_name in enumerate(class_names):
        if i == class_idx:
            plt.plot(time_points, shap_vals[:, i], label=f"SHAP Values (Class {class_name})", 
                     color=colors[i], linewidth=3)
        else:
            plt.plot(time_points, shap_vals[:, i], label=f"SHAP Values (Class {class_name})", 
                     color=colors[i], alpha=0.5)
    
    # Add labels and title
    plt.xlabel("Time Point")
    plt.ylabel("Value")
    plt.title(f"SHAP Values for ECG Time-Series (Class {class_names[class_idx]})")
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage
if __name__ == "__main__":
    # Example input: random array with shape (1, 187) or (187,)
    # x_instance = np.random.rand(1, 187)  # Replace with actual unscaled test instance
    import pandas as pd
    df_test = pd.read_csv('mitbih_test.csv')  # Replace with actual test data file
    df_test = df_test.dropna()
    x_instance = df_test.iloc[20002, :-1].values  # First instance, excluding label
    try:
        shap_vals, x_instance_scaled = get_shap_values_for_instance(x_instance)
        print("SHAP values shape:", shap_vals.shape)  # Output: (187, 5)
        plot_shap_values(x_instance_scaled, shap_vals)
    except ValueError as e:
        print(f"Error: {e}")