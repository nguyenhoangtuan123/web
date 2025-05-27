from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from scipy.special import softmax
import numpy as np
import scipy.optimize as opt
import xgboost as xgb
from shap_xgboost import get_shap_values_for_instance
from cvxopt import matrix, solvers
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, ClassifierMixin
from flask import Flask, request, jsonify
import joblib
import pandas as pd
from flask_cors import CORS
import os
import logging
import asyncio
import functools
from concurrent.futures import ThreadPoolExecutor
from cachetools import TTLCache

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True, methods=["GET", "POST", "OPTIONS"])

# Cache để lưu dữ liệu mẫu (TTL = 1 giờ)
sample_cache = TTLCache(maxsize=1, ttl=3600)
executor = ThreadPoolExecutor(max_workers=4)  # Thread pool cho xử lý bất đồng bộ

# Định nghĩa các tham số mô hình và scaler
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
    'SVC_classweight',
]

# Các class mô hình (giữ nguyên từ code của bạn)
class MyLogisticRegression:
    def __init__(self, C=1.0):
        self.C = C
        self.coef_ = None

    def _softmax(self, X, beta):
        beta = beta.reshape(self.n_classes_, self.n_features_)
        logits = X @ beta.T
        return softmax(logits, axis=1)

    def _log_likelihood(self, beta, X, y):
        beta = beta.reshape(self.n_classes_, self.n_features_)
        logits = X @ beta.T
        log_probs = logits - np.log(np.sum(np.exp(logits), axis=1, keepdims=True))
        reg_term = (self.C / 2) * np.sum(beta[:, 1:] ** 2)
        return -np.sum(y * log_probs) + reg_term

    def _likelihood_gradient(self, beta, X, y):
        beta = beta.reshape(self.n_classes_, self.n_features_)
        probs = self._softmax(X, beta)
        reg_grad = self.C * beta[:, 1:]
        reg_grad = np.concatenate([np.zeros((beta.shape[0], 1)), reg_grad], axis=1)
        return ((X.T @ (probs - y)).T + reg_grad).flatten()

    def fit(self, X, y):
        y = y.astype(int)
        X = np.hstack([np.ones((X.shape[0], 1)), X])
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
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        return X @ self.coef_.T

    def predict_proba(self, X):
        return self._softmax(X, self.coef_)

    def predict(self, X):
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        return np.argmax(self.predict_proba(X), axis=1)

    def get_coef(self):
        return self.coef_

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

class SVMQP(BaseEstimator, ClassifierMixin):
    def __init__(self, epsilon=1e-5, C=100, kernel='linear', gamma=0.02, class_weight=None):
        self.lambdas = None
        self.epsilon = epsilon
        self.C = C
        assert kernel in ['linear', 'rbf'], "Invalid kernel"
        self.kernel = kernel
        self.gamma = gamma
        self.class_weight = class_weight

    def fit(self, X, y):
        if self.gamma == 'scale':
            n_features = X.shape[1]
            var_X = X.var()
            self._gamma = 1.0 / (n_features * var_X) if var_X > 0 else 1.0
        else:
            self._gamma = self.gamma
        self.X = np.array(X)
        self.y = np.array(2 * y - 1).astype(np.double)
        N = self.X.shape[0]
        V = self.X * np.expand_dims(self.y, axis=1)
        unique_classes, class_counts = np.unique(y, return_counts=True)
        num_classes = len(unique_classes)
        if self.class_weight == 'balanced':
            class_weight_dict = {cls: N / (num_classes * count) for cls, count in zip(unique_classes, class_counts)}
        elif isinstance(self.class_weight, np.ndarray):
            assert len(self.class_weight) == num_classes, "class_weight size must match number of classes"
            class_weight_dict = {cls: weight for cls, weight in zip(unique_classes, self.class_weight)}
        else:
            class_weight_dict = {cls: 1.0 for cls in unique_classes}
        sample_weights = np.array([class_weight_dict[yi] for yi in y])
        if self.kernel == 'rbf':
            K = matrix(np.outer(self.y, self.y) * self.rbf_kernel(self.X, self.X))
        else:
            K = matrix(V.dot(V.T))
        p = matrix(-np.ones((N, 1)))
        G = matrix(np.vstack((-np.eye(N), np.eye(N))))
        h = matrix(np.vstack((np.zeros((N, 1)), (self.C * sample_weights).reshape(N, 1))))
        A = matrix(self.y.reshape(-1, N))
        b = matrix(np.zeros((1, 1)))
        solvers.options['show_progress'] = False
        sol = solvers.qp(K, p, G, h, A, b)
        self.lambdas = np.array(sol['x'])
        self.get_wb()

    def rbf_kernel(self, X1, X2):
        sq_dists = cdist(X1, X2, 'sqeuclidean')
        return np.exp(-1.0 * self._gamma * sq_dists)

    def get_lambdas(self):
        return self.lambdas

    def get_wb(self):
        S = np.where(self.lambdas > self.epsilon)[0]
        V = self.X * np.expand_dims(self.y, axis=1)
        VS = V[S, :]
        XS = self.X[S, :]
        yS = self.y[S]
        lS = self.lambdas[S]
        self.XS = XS
        self.yS = yS
        self.lS = lS
        if self.kernel == 'rbf':
            alpha = lS * np.expand_dims(yS, axis=1)
            b = np.mean(np.expand_dims(yS, axis=1) - self.rbf_kernel(XS, XS).dot(alpha))
            self.b = b
            return b
        else:
            w = lS.T.dot(VS)
            b = np.mean(np.expand_dims(yS, axis=1) - XS.dot(w.T))
            self.w = w
            self.b = b
            return self.w, self.b

    def predict(self, X_test):
        K_test = self.rbf_kernel(X_test, self.XS)
        conf = K_test @ (self.lS * np.expand_dims(self.yS, axis=1)) + self.b
        return (np.squeeze(np.sign(conf)) + 1) // 2

    def decision_function(self, X_test):
        K_test = self.rbf_kernel(X_test, self.XS)
        conf = K_test @ (self.lS * np.expand_dims(self.yS, axis=1)) + self.b
        return np.squeeze(conf)

# Tải và lưu scaler
def process_mitbih():
    try:
        logger.info("Loading mitbih_train.csv for scaler...")
        df = pd.read_csv("mitbih_train", header=None)
        X = df.iloc[:, :-1].values
        scaler = StandardScaler()
        scaler.fit(X)
        scaler_filename = "scaler.pkl"
        joblib.dump(scaler, scaler_filename)
        logger.info(f"StandardScaler saved to {scaler_filename}")
        return scaler
    except Exception as e:
        logger.error(f"Error processing mitbih data: {str(e)}")
        return None

def process_mitbih_fft():
    try:
        logger.info("Loading mitbih_train.csv for FFT scaler...")
        df = pd.read_csv("mitbih_train", header=None)
        X = df.iloc[:, :-1].values
        X_fft = np.fft.fft(X, axis=1)
        X_fft_magnitude = np.abs(X_fft[:, :X_fft.shape[1] // 2])
        fft_scaler = StandardScaler()
        fft_scaler.fit(X_fft_magnitude)
        scaler_filename = "fft_scaler.pkl"
        joblib.dump(fft_scaler, scaler_filename)
        logger.info(f"FFT StandardScaler saved to {scaler_filename}")
        return fft_scaler
    except Exception as e:
        logger.error(f"Error processing mitbih data for FFT: {str(e)}")
        return None

try:
    fft_scaler = joblib.load("fft_scaler.pkl")
    logger.info("Successfully loaded fft_scaler.pkl")
except FileNotFoundError:
    logger.warning("fft_scaler.pkl not found, creating new one...")
    fft_scaler = process_mitbih_fft()

try:
    scaler = joblib.load("scaler.pkl")
    logger.info("Successfully loaded scaler.pkl")
except FileNotFoundError:
    logger.warning("scaler.pkl not found, creating new one...")
    scaler = process_mitbih()

# Tải các mô hình
models = []
for model_name in model_names:
    try:
        model = joblib.load(f"{model_name}.pkl")
        logger.info(f"Successfully loaded {model_name}.pkl")
        models.append(model)
    except Exception as e:
        logger.error(f"Error loading {model_name}.pkl: {str(e)}")
        models.append(None)

# Khởi tạo sample_data khi khởi động
def init_sample_data():
    try:
        logger.info("Initializing sample data...")
        df = pd.read_csv("mitbih_train", header=None, on_bad_lines='skip')
        df = df.dropna()
        if df.empty:
            raise ValueError("No valid data in mitbih_train.csv")
        label_col = df.columns[-1]
        class_counts = df[label_col].value_counts()
        if len(class_counts) < 5:
            raise ValueError("Less than 5 classes found in the data")
        samples_per_class = 30 // 5
        sampled_df = df.groupby(label_col).apply(
            lambda g: g.sample(n=min(samples_per_class, len(g)), random_state=42)
        ).reset_index(drop=True)
        if len(sampled_df) < 30:
            needed = 30 - len(sampled_df)
            extras = df[df[label_col].isin(class_counts[class_counts > samples_per_class].index)]
            extra_samples = extras.sample(n=needed, random_state=42)
            sampled_df = pd.concat([sampled_df, extra_samples], ignore_index=True)
        sample_cache['data'] = sampled_df.values
        logger.info(f"Sampled {len(sample_cache['data'])} rows with all 5 classes")
    except Exception as e:
        logger.error(f"Error initializing sample data: {e}")
        sample_cache['data'] = np.array([])

# Chạy khởi tạo sample_data
init_sample_data()

# Hàm xử lý bất đồng bộ
def run_in_executor(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(executor, lambda: func(*args, **kwargs))
    return wrapper

# Endpoint để lấy dữ liệu mẫu
@app.route("/get_samples", methods=["GET"])
async def get_samples():
    logger.info("Starting get_samples endpoint")
    try:
        if 'data' not in sample_cache or not sample_cache['data'].size:
            return jsonify({"error": "No sample data available"}), 404
        samples = []
        labels = ["Normal", "Supraventricular Ectopic", "Ventricular Ectopic", "Fusion", "Unknown"]
        for i, row in enumerate(sample_cache['data']):
            if len(row) < 188:
                continue
            signal = row[:-1]
            if not all(isinstance(x, (int, float)) for x in signal):
                logger.warning(f"Invalid signal data in row {i}")
                continue
            try:
                label = int(row[-1])
            except ValueError:
                logger.warning(f"Invalid label in row {i}: {row[-1]}")
                continue
            samples.append({
                "id": i,
                "label": labels[label],
                "signal": signal.tolist()
            })
        if not samples:
            return jsonify({"error": "No valid sample data available"}), 404
        logger.info("Finished get_samples endpoint")
        return jsonify({"samples": samples})
    except Exception as e:
        logger.error(f"Error in get_samples: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Hàm dự đoán bất đồng bộ cho một mô hình
@run_in_executor
def predict_with_model(model, model_name, data, fft_data, scaler, fft_scaler):
    labels = ["Normal", "Supraventricular Ectopic", "Ventricular Ectopic", "Fusion", "Unknown"]
    try:
        if model_name == "XGBoost_SMOTE" or model_name == "XGBoost_classweight":
            Ddata_scaled = xgb.DMatrix(data)
            prediction = model.predict(Ddata_scaled)[0]
        elif model_name == "XGBoost_FFT":
            Dfft_data = xgb.DMatrix(fft_data)
            prediction = model.predict(Dfft_data)[0]
        elif model_name.endswith("FFT"):
            prediction = model.predict(fft_data)[0]
        else:
            prediction = model.predict(data)[0]
        pred_int = int(prediction)
        if pred_int < 0 or pred_int >= len(labels):
            raise ValueError(f"Prediction {pred_int} out of range for labels")
        result = labels[pred_int]
        try:
            if model_name == "LogisticRegression_scratch":
                data_scaled_tmp = np.hstack([np.ones((data.shape[0], 1)), data])
                probabilities = model.predict_proba(data_scaled_tmp)[0].tolist()
            elif model_name == "XGBoost_SMOTE" or model_name == "XGBoost_classweight":
                probabilities = model.predict_proba(xgb.DMatrix(data))[0].tolist()
            elif model_name == "SVC_scratch" or model_name == "SVC_SMOTE" or model_name == "SVC_classweight":
                probabilities = model.decision_function(data)[0].tolist()
                probabilities = softmax(probabilities).tolist()
            elif model_name == "XGBoost_FFT":
                probabilities = model.predict_proba(xgb.DMatrix(fft_data))[0].tolist()
            elif model_name == "SVC_FFT":
                probabilities = model.decision_function(fft_data)[0].tolist()
                probabilities = softmax(probabilities).tolist()
            elif model_name.endswith("FFT"):
                probabilities = model.predict_proba(fft_data)[0].tolist()
            else:
                probabilities = model.predict_proba(data)[0].tolist()
            return {
                "model": model_name,
                "prediction": result,
                "probabilities": probabilities
            }
        except AttributeError:
            logger.warning(f"Model {model_name} does not support predict_proba")
            return {
                "model": model_name,
                "prediction": result,
                "error": "Model does not support predict_proba"
            }
    except Exception as e:
        logger.error(f"Error with model {model_name}: {str(e)}")
        return {
            "model": model_name,
            "error": str(e)
        }

# Endpoint để phân loại tín hiệu
@app.route('/classify', methods=['POST'])
async def classify():
    logger.info("Starting classify endpoint")
    try:
        data = request.json.get('signal')
        model_name = request.json.get('model_name')  # Thêm tham số để chọn mô hình
        if not data or not isinstance(data, list):
            return jsonify({"error": "Invalid signal data"}), 400
        if len(data) != 187:
            return jsonify({"error": "Signal must have length 187"}), 400
        if not all(isinstance(x, (int, float)) for x in data):
            return jsonify({"error": "Signal must contain only numbers"}), 400
        if scaler is None or fft_scaler is None:
            return jsonify({"error": "Scaler not loaded"}), 500
        if model_name not in model_names:
            return jsonify({"error": "Invalid model name"}), 400

        data = np.array(data).reshape(1, -1)
        try:
            data_scaled = scaler.transform(data)
            X_fft = np.fft.fft(data, axis=1)
            X_fft_magnitude = np.abs(X_fft[:, :X_fft.shape[1] // 2])
            fft_data = fft_scaler.transform(X_fft_magnitude)
        except Exception as e:
            logger.error(f"Scaler error with data shape {data.shape}: {str(e)}")
            return jsonify({"error": f"Scaler transformation failed: {str(e)}"}), 500

        model_idx = model_names.index(model_name)
        model = models[model_idx]
        if model is None:
            return jsonify({"error": f"Model {model_name} not loaded"}), 500

        result = await predict_with_model(model, model_name, data_scaled, fft_data, scaler, fft_scaler)
        logger.info("Finished classify endpoint")
        return jsonify({"results": [result]})
    except Exception as e:
        logger.error(f"Error in classify endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 400

# Endpoint để lấy SHAP values
@app.route('/get_shape_xgboost', methods=['POST'])
async def get_shape_xgboost():
    logger.info("Starting get_shape_xgboost endpoint")
    try:
        data = request.json.get("signal")
        if not data:
            return jsonify({"error": "No data provided"}), 400
        shap_values, scaled_instance, class_idx = get_shap_values_for_instance(data)
        shap_values = np.transpose(shap_values)
        logger.info("Finished get_shape_xgboost endpoint")
        return jsonify({
            "shap_values": shap_values.tolist(),
            "scaled_instance": scaled_instance.tolist(),
            "class_idx": int(class_idx)
        })
    except Exception as e:
        logger.error(f"Error in get_shape_xgboost endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)