from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier(
    n_estimators=3,  # Number of boosting stages
    learning_rate=0.05,  # Smaller values help prevent overfitting
    max_depth=5,  # Controls tree depth
    subsample=0.8,  # Helps generalization
    random_state=42
)
import numpy as np
import scipy.optimize as opt
from sklearn.linear_model import LogisticRegression
from scipy.special import softmax
import xgboost as xgb
from shap_xgboost import get_shap_values_for_instance

class MyLogisticRegression:
    def __init__(self, C=1.0):
        self.C = C
        self.coef_ = None

    def _softmax(self, X, beta):
        beta = beta.reshape(self.n_classes_, self.n_features_)
        logits = X @ beta.T
        return softmax(logits,axis=1)

    def _log_likelihood(self, beta, X, y):
        beta = beta.reshape(self.n_classes_, self.n_features_)
        logits = X @ beta.T
        log_probs = logits - np.log(np.sum(np.exp(logits), axis=1, keepdims=True))
        reg_term = (self.C / 2) * np.sum(beta[:,1:] ** 2)  # L2 regularization term
        return -np.sum(y * log_probs) + reg_term

    def _likelihood_gradient(self, beta, X, y):
        beta = beta.reshape(self.n_classes_, self.n_features_)
        probs = self._softmax(X, beta)
        reg_grad = self.C * beta[:,1:] # L2 
        reg_grad = np.concatenate([np.zeros((beta.shape[0],1)),reg_grad],axis=1)
        return ((X.T @ (probs - y)).T+reg_grad).flatten()
        
    def fit(self, X, y):
        y = y.astype(int)
        X = np.hstack([np.ones((X.shape[0], 1)), X])  # Add intercept
        self.n_classes_ = len(np.unique(y))
        self.n_features_ = X.shape[1]
        
        y_one_hot = np.eye(self.n_classes_)[y]
        self.y_one_hot = y_one_hot
        beta_init = np.zeros((self.n_classes_, self.n_features_)).flatten()
        
        result = opt.minimize(
            self._log_likelihood,
            beta_init,
            args=(X, y_one_hot),
            method='BFGS',
            jac=self._likelihood_gradient,
            options={'disp': False}
        )
        
        self.coef_ = result.x.reshape(self.n_classes_, self.n_features_)
        return self

    def decision_function(self, X):
        X = np.hstack([np.ones((X.shape[0], 1)), X])  # Add intercept
        return X @ self.coef_.T

    def predict_proba(self, X):
        return self._softmax(X, self.coef_)

    def predict(self, X):
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        return np.argmax(self.predict_proba(X), axis=1)

    def get_coef(self):
        return self.coef_

from cvxopt import matrix, solvers
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, ClassifierMixin

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def sigmoid(z):
    return 1/(1 + np.exp(-z))

class SVMQP(BaseEstimator, ClassifierMixin):
    def __init__(self, epsilon=1e-5, C=100, kernel='linear', gamma=0.02, class_weight=None):
        self.lambdas = None
        self.epsilon = epsilon
        self.C = C
        assert kernel in ['linear', 'rbf'], "broooooo valid kernel please"
        self.kernel = kernel
        self.gamma = gamma # for kernel = 'rbf'
        self.class_weight = class_weight # default class_weight=None
    
    def fit(self, X, y):
        if self.gamma == 'scale': # like sklearn
            n_features = X.shape[1]
            var_X = X.var()
            self._gamma = 1.0 / (n_features*var_X) if var_X > 0 else 1.0
            # print('gamma:',self.gamma)
        else: self._gamma = self.gamma
            
        self.X = np.array(X)
        self.y = np.array(2*y-1).astype(np.double)
        N = self.X.shape[0] 
        V = self.X*np.expand_dims(self.y, axis=1)

        # Compute class weights for 'balanced' mode
        unique_classes, class_counts = np.unique(y, return_counts=True)
        num_classes = len(unique_classes)
        if self.class_weight == 'balanced':
            # Calculate weights inversely proportional to class frequencies
            class_weight_dict = {cls: N / (num_classes * count) for cls, count in zip(unique_classes, class_counts)}
        elif isinstance(self.class_weight, np.ndarray):
            assert len(self.class_weight) == num_classes, "class_weight size must match number of classes"
            class_weight_dict = {cls: weight for cls, weight in zip(unique_classes, self.class_weight)}
        else:
            # Default: equal weights
            class_weight_dict = {cls: 1.0 for cls in unique_classes}

        # Assign sample weights based on class weights
        sample_weights = np.array([class_weight_dict[yi] for yi in y])

        if (self.kernel=='rbf'):
            K = matrix(np.outer(self.y, self.y)*self.rbf_kernel(self.X, self.X))
        else:
            K = matrix(V.dot(V.T))

        # print("K's determinant is:", np.linalg.det(K))
        
        # print(K.shape)
        # K = matrix(V.dot(V.T))
        # print(K.shape)
        
        p = matrix(-np.ones((N, 1)))
        G = matrix(np.vstack((-np.eye(N), np.eye(N))))
        h = matrix(np.vstack((np.zeros((N, 1)), (self.C * sample_weights).reshape(N, 1))))
        A = matrix(self.y.reshape(-1, N))
        b = matrix(np.zeros((1, 1)))
        
        solvers.options['show_progress'] = False
        # print(X, y)

        print('solving QP')
        sol = solvers.qp(K, p, G, h, A, b)
        self.lambdas = np.array(sol['x'])
        self.get_wb()
        
    def rbf_kernel(self, X1, X2):
        """Compute the RBF kernel matrix."""
        sq_dists = cdist(X1, X2, 'sqeuclidean')
        return np.exp(-1.0*self._gamma * sq_dists)
        
    def get_lambdas(self):
        return self.lambdas

    def get_wb(self):
        S = np.where(self.lambdas > self.epsilon)[0]
        V = self.X*np.expand_dims(self.y, axis=1)

        VS = V[S, :]
        XS = self.X[S, :]
        yS = self.y[S]
        lS = self.lambdas[S]

        self.XS = XS
        self.yS = yS
        self.lS = lS
        
        if self.kernel == 'rbf':
            alpha = lS*np.expand_dims(yS,axis=1)
            b = np.mean(np.expand_dims(yS,axis=1) - self.rbf_kernel(XS, XS).dot(alpha))
            self.b  = b

            return b
            # b = np.mean()
        else:
            w =  lS.T.dot(VS)
            b = np.mean(np.expand_dims(yS,axis=1) - XS.dot(w.T))
            self.w = w
            self.b = b
            
            return self.w, self.b
    
    def print_lambdas(self):
        print('lambda = ')
        print(self.lambdas.T)
        S = np.where(self.lambdas > self.epsilon)[0]
        print(self.lambdas[S])

    def predict(self, X_test):
        K_test = self.rbf_kernel(X_test, self.XS)
        conf = K_test@(self.lS*np.expand_dims(self.yS,axis=1))+self.b
        return (np.squeeze(np.sign(conf))+1)//2

    def decision_function(self, X_test):
        K_test = self.rbf_kernel(X_test, self.XS)
        conf = K_test@(self.lS*np.expand_dims(self.yS,axis=1))+self.b
        return np.squeeze(conf)
    
    def predict_proba(self, X_test):
        assert False, "my class is not random"
        K_test = self.rbf_kernel(X_test, self.XS)
        conf = K_test@(self.lS*np.expand_dims(self.yS,axis=1))+self.b

from flask import Flask, request, jsonify
import joblib
import pandas as pd
from flask_cors import CORS
import os
# from pymongo import MongoClient
# import datetime

# # Kết nối tới MongoDB local (mặc định port 27017)
# client = MongoClient("mongodb://localhost:27017/")
# db = client["ecg_database"]           # Tên database
# results_collection = db["results"]    # Tên collection

app = Flask(__name__)

CORS(app)  # Allow CORS from all origins

print("Current directory:", os.getcwd())

# Define model filenames (REPLACE WITH YOUR ACTUAL FILENAMES)
model_names = [
            'LogisticRegression_scratch', 
            'SVC_scratch', 
            'GradientBoostingClassifier_SMOTE',
            'XGBoost_SMOTE',
            'SVC_SMOTE', 
            'LogisticRegression_SMOTE',
            'GradientBoostingClassifier_FFT',
            'XGBoost_FFT', 
            'LogisticRegression_FFT', 
            'SVC_FFT',
            'GradientBoostingClassifier_classweight',
            'XGBoost_classweight',
            'LogisticRegression_classweight',
            'SVC_classweight',]

# Load scaler
# try:
#     scaler = joblib.load("scaler.pkl")
#     print("Successfully loaded scaler.pkl")
# except FileNotFoundError:
#     print("Error: Could not find scaler.pkl")
#     scaler = None
# except Exception as e:
#     print(f"Error loading scaler.pkl: {str(e)}")
#     scaler = None

# get mitbih_train.csv convert it to fft then save the standard scaler
def process_mitbih_fft():
    """
    Process the mitbih_train.csv file by applying FFT and saving a standard scaler
    """
    try:
        # Load data
        print("Loading mitbih_train.csv...")
        df = pd.read_csv("mitbih_train.csv", header=None)
        print(f"Loaded dataset with shape: {df.shape}")
        
        # Extract features (all columns except the last one which is the target)
        X = df.iloc[:, :-1].values
        
        # Apply FFT
        print("Applying FFT transformation...")
        X_fft = np.fft.fft(X, axis=1)
        
        # Use magnitude of FFT (discard phase information)
        # Only use first half as the second half is symmetric for real signals
        X_fft_magnitude = np.abs(X_fft[:, :X_fft.shape[1] // 2])
        
        
        # Apply standard scaling
        print("Fitting standard scaler...")
        fft_scaler = StandardScaler()
        fft_scaler.fit(X_fft_magnitude)
        print("Shape of FFT data:", X_fft_magnitude.shape)
        
        # Save the scaler
        scaler_filename = "fft_scaler.pkl"
        joblib.dump(fft_scaler, scaler_filename)
        print(f"StandardScaler saved to {scaler_filename}")
        
        return fft_scaler
    except FileNotFoundError:
        print("Error: mitbih_train.csv not found")
        return None
    except Exception as e:
        print(f"Error processing mitbih data: {str(e)}")
        return None
try:
    fft_scaler = joblib.load("fft_scaler.pkl")
    print("Successfully loaded fft_scaler.pkl")
except FileNotFoundError:
    print("fft_scaler.pkl not found, creating new one...")
    fft_scaler = process_mitbih_fft()

def process_mitbih():
    """
    Process the mitbih_train.csv file and saving a standard scaler
    """
    try:
        # Load data
        print("Loading mitbih_train.csv...")
        df = pd.read_csv("mitbih_train.csv", header=None)
        print(f"Loaded dataset with shape: {df.shape}")
        
        # Extract features (all columns except the last one which is the target)
        X = df.iloc[:, :-1].values
        
        scaler = StandardScaler()
        scaler.fit(X)
        
        # Save the scaler
        scaler_filename = "scaler.pkl"
        joblib.dump(scaler, scaler_filename)
        print(f"StandardScaler saved to {scaler_filename}")

        return scaler
    except FileNotFoundError:
        print("Error: mitbih_train.csv not found")
        return None
    except Exception as e:
        print(f"Error processing mitbih data: {str(e)}")
        return None
try:
    scaler = joblib.load("scaler.pkl")
    print("Successfully loaded scaler.pkl")
except FileNotFoundError:
    print("scaler.pkl not found, creating new one...")
    scaler = process_mitbih()

# Load models
models = []
for model_name in model_names:
    try:
        model = joblib.load(f"{model_name}.pkl")
        print(f"Successfully loade  d {model_name}.pkl")
        models.append(model)
    except FileNotFoundError:
        print(f"Error: Could not find {model_name}.pkl")
        models.append(None)
    except Exception as e:
        print(f"Error loading {model_name}.pkl: {str(e)}")
        models.append(None)

# Load sample data from mitbih_train.csv with error-handling
try:
    df = pd.read_csv("mitbih_train.csv", header=None, on_bad_lines='skip')
    print(f"Loaded {len(df)} rows from mitbih_train.csv")

    df = df.dropna()
    print(f"After dropna: {len(df)} rows")

    if not df.empty:
        # Assume the label is in the last column
        label_col = df.columns[-1]
        class_counts = df[label_col].value_counts()
        
        if len(class_counts) < 5:
            raise ValueError("Less than 5 classes found in the data")

        # Determine number of samples per class (e.g. 6 for 5 classes = 30 total)
        samples_per_class = 30 // 5

        # Collect samples
        sampled_df = df.groupby(label_col).apply(
            lambda g: g.sample(n=min(samples_per_class, len(g)), random_state=42)
        ).reset_index(drop=True)

        # If total < 30 due to class imbalance, optionally sample more from over-represented classes
        if len(sampled_df) < 30:
            needed = 30 - len(sampled_df)
            extras = df[df[label_col].isin(class_counts[class_counts > samples_per_class].index)]
            extra_samples = extras.sample(n=needed, random_state=42)
            sampled_df = pd.concat([sampled_df, extra_samples], ignore_index=True)

        sample_data = sampled_df.values
        print(f"Sampled {len(sample_data)} rows with all 5 classes")
    else:
        sample_data = np.array([])
        print("Error: mitbih_train.csv contains no valid data")

except FileNotFoundError:
    sample_data = np.array([])
    print("Error: mitbih_train.csv not found")
except Exception as e:
    sample_data = np.array([])
    print(f"Error loading data: {e}")

# Endpoint to get sample data
@app.route("/get_samples", methods=["GET"])
def get_samples():
    if not sample_data.size:
        return jsonify({"error": "No sample data available"}), 404
    
    samples = []
    for i, row in enumerate(sample_data):
        if len(row) < 188:
            continue
        signal = row[:-1]
        if not all(isinstance(x, (int, float)) for x in signal):
            print(f"Invalid signal data in row {i}")
            continue
        try:
            label = int(row[-1])
        except ValueError:
            print(f"Invalid label in row {i}: {row[-1]}")
            continue
        labels = ["Normal", "Supraventricular Ectopic", "Ventricular Ectopic", "Fusion", "Unknown"]
        samples.append({
            "id": i,
            "label": labels[label],
            "signal": signal.tolist()
        })
    if not samples:
        return jsonify({"error": "No valid sample data available"}), 404
    return jsonify({"samples": samples})

# @app.route("/save-result", methods=["POST"])
# def save_result():
#     data = request.get_json()
#     result = data.get("result")

#     if result is None:
#         return jsonify({"error": "Missing result field"}), 400

#     # Ghi vào MongoDB
#     results_collection.insert_one({
#         "result": result,
#         "timestamp": datetime.datetime.now()
#     })

#     return jsonify({"message": "Result saved successfully"})
# Endpoint to classify signal with all models
@app.route('/classify', methods=['POST'])
def classify():
    try:
        data = request.json.get('signal')
        if not data or not isinstance(data, list):
            return jsonify({"error": "Invalid signal data"}), 400
        
        if len(data) != 187:
            return jsonify({"error": "Signal must have length 187"}), 400
        
        if not all(isinstance(x, (int, float)) for x in data):
            return jsonify({"error": "Signal must contain only numbers"}), 400
        
        if scaler is None:
            return jsonify({"error": "Scaler not loaded"}), 500

        data = np.array(data).reshape(1, -1)
        labels = ["Normal", "Supraventricular Ectopic", "Ventricular Ectopic", "Fusion", "Unknown"]
        results = []
        
        # Apply scaler once
        try:
            data_scaled = scaler.transform(data)
            X_fft = np.fft.fft(data, axis=1)
            X_fft_magnitude = np.abs(X_fft[:, :X_fft.shape[1] // 2])
            fft_data = fft_scaler.transform(X_fft_magnitude)
            print(data_scaled)
        except Exception as e:
            print(f"Scaler error with data shape {data.shape}: {str(e)}")
            return jsonify({"error": f"Scaler transformation failed: {str(e)}"}), 500

        for i, (model, model_name) in enumerate(zip(models, model_names)):
            if model is None:
                results.append({
                    "model": model_name,
                    "error": f"Model {model_name} not loaded"
                })
                continue
            
            try:
                if model_name == "XGBoost_SMOTE" or model_name == "XGBoost_classweight":
                    Ddata_scaled = xgb.DMatrix(data_scaled)
                    prediction = model.predict(Ddata_scaled)[0]
                elif model_name == "XGBoost_FFT":
                    Dfft_data = xgb.DMatrix(fft_data)
                    prediction = model.predict(Dfft_data)[0]
                elif model_name.endswith("FFT"):
                    prediction = model.predict(fft_data)[0]
                else:
                    prediction = model.predict(data_scaled)[0]
                try:
                    pred_int = int(prediction)
                    if pred_int < 0 or pred_int >= len(labels):
                        raise ValueError(f"Prediction {pred_int} out of range for labels")
                    result = labels[pred_int]
                except (ValueError, TypeError) as e:
                    results.append({
                        "model": model_name,
                        "error": f"Invalid prediction: {str(e)}"
                    })
                    continue
                
                try:
                    print(model_name)
                    if model_name=="LogisticRegression_scratch":
                        data_scaled_tmp = np.hstack([np.ones((data_scaled.shape[0], 1)), data_scaled])
                        probabilities = model.predict_proba(data_scaled_tmp)[0].tolist()
                    elif model_name=="XGBoost_SMOTE" or model_name=="XGBoost_classweight":
                        probabilities = model.predict_proba(xgb.DMatrix(data_scaled, label = np.random.randint(0,5,data_scaled.shape[1]))
                                                            )[0].tolist()
                    elif model_name=="SVC_scratch" or model_name=="SVC_SMOTE" or model_name=="SVC_classweight":
                        probabilities = model.decision_function(data_scaled)[0].tolist()
                        # use np softmax to convert to probabilities
                        probabilities = softmax(probabilities).tolist()
                    elif model_name=="XGBoost_FFT":
                        probabilities = model.predict_proba(xgb.DMatrix(fft_data, label = np.random.randint(0,5,fft_data.shape[1]))
                                                            )[0].tolist()
                    elif model_name=="SVC_FFT":
                        probabilities = model.decision_function(fft_data)[0].tolist()
                        probabilities = softmax(probabilities).tolist()
                    elif model_name.endswith("FFT"):
                        probabilities = model.predict_proba(fft_data)[0].tolist()
                    else: probabilities = model.predict_proba(data_scaled)[0].tolist()
                    results.append({
                        "model": model_name,
                        "prediction": result,
                        "probabilities": probabilities
                    })
                except AttributeError:
                    print(f"Model {model_name} does not support predict_proba")
                    results.append({
                        "model": model_name,
                        "prediction": result,
                        "error": "Model does not support predict_proba"
                    })
            except Exception as e:
                print(f"Error with model {model_name}: {str(e)}")
                results.append({
                    "model": model_name,
                    "error": str(e)
                })
        
        return jsonify({"results": results})
    except Exception as e:
        print(f"Error in classify endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 400

@app.route('/get_shape_xgboost', methods=['POST'])
def get_shape_xgboost():
    try:
        data = request.json.get("signal")
        if not data:
            return jsonify({"error": "No data provided"}), 400

        shap_values, scaled_instance, class_idx = get_shap_values_for_instance(data)
        # transpose shap_values
        shap_values = np.transpose(shap_values) # shape = (5, 187)
        # scaled_instance = shape (187, )
        return jsonify({"shap_values": shap_values.tolist(),
            "scaled_instance": scaled_instance.tolist(),
            "class_idx": int(class_idx)})
    except Exception as e:
        print(f"Error in get_shape_xgboost endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))