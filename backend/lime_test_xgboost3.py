import xgboost as xgb
import lime.lime_tabular
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

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
X_test = scaler.transform(X_test)

class_names = ["Normal", "Supraventricular", "Ventricular", "Fusion", "Unknown"]

# Load the XGBoost Booster model
model = joblib.load('XGBoost_softprob_for_LIME.pkl')

def predict_proba_wrapper(X):
    dmatrix = xgb.DMatrix(X)
    return model.predict(dmatrix)  # Ensure model was trained with 'multi:softprob'

# Verify model prediction
print(predict_proba_wrapper(X_test[20000:20001]))

# LIME explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=[str(i) for i in range(187)],  # Convert feature names to strings
    class_names=class_names,
    mode='classification'
)

# Explain one instance
i = 0
exp = explainer.explain_instance(X_test[i], predict_proba_wrapper, num_features=187, top_labels=5)

# Convert explanation data to JSON-serializable format
def convert_to_serializable(exp):
    serializable_exp = []
    for label, feature_weights in exp.local_exp.items():
        serializable_weights = [
            (int(feature), float(weight))  # Convert feature index (int32) to int, weight to float
            for feature, weight in feature_weights
        ]
        serializable_exp.append((label, serializable_weights))
    return serializable_exp

# Override the local_exp attribute with serializable data
exp.local_exp = dict(convert_to_serializable(exp))

# Extract and print feature values for each class
print(f"Explanation for instance {i}:")
for class_idx, feature_weights in exp.local_exp.items():
    class_name = class_names[class_idx]
    print(f"\nClass: {class_name} (Index: {class_idx})")
    print("Feature Index | Feature Name | Weight")
    print("-" * 40)
    for feature_idx, weight in feature_weights:
        feature_name = f"Time Point {feature_idx}"
        print(f"{feature_idx:11} | {feature_name:12} | {weight:.6f}")

# Optionally, store the results in a dictionary
feature_importance = {
    class_names[class_idx]: [
        {"feature": f"Time Point {feat_idx}", "weight": weight}
        for feat_idx, weight in feature_weights
    ]
    for class_idx, feature_weights in exp.local_exp.items()
}

# Print the dictionary (optional)
import json
print("\nFeature Importance Dictionary:")
print(json.dumps(feature_importance, indent=2))

# Save the explanation to HTML
# html_path = "lime_explanation.html"
# exp.save_to_file(html_path)

# print(f"Explanation saved to {html_path}")