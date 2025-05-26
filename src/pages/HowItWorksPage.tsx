import React, { useRef } from 'react';

const reportContent = {
  introduction: {
    problem: "Analyzing electrocardiogram (ECG) data is crucial in diagnosing heart diseases, but this process is time-consuming and requires high expertise when done manually. The problem posed is to build an automated system for classifying heartbeats using machine learning to support faster and more accurate diagnosis. This project explores and compares the effectiveness of three popular classification algorithms: Support Vector Machine (SVM), Logistic Regression, and Gradient Boosting (XGBoost) on real heartbeat data, including both using libraries and reimplementing these algorithms from scratch.",
    dataset: "The dataset used in this project is the heartbeat dataset downloaded from /kaggle/input/heartbeat/mitbih_train.csv. This is a popular dataset in ECG signal classification problems, often associated with the MIT-BIH Arrhythmia Database. The initial dataset contains 87,553 data samples and 188 attributes. All attributes are of type float64. Among them, the first 187 attributes represent data points on the ECG signal, and the last attribute (column 188) is the target variable, indicating the corresponding heartbeat type.",
    objectives: [
      "Apply and evaluate classification algorithms: Implement and evaluate the effectiveness of three machine learning algorithms: Support Vector Machine (SVM), Logistic Regression, and Gradient Boosting (with the XGBoost variant) on real heartbeat data.",
      "Compare two implementation methods: Implement and use the algorithms in two ways: using existing machine learning libraries (such as scikit-learn, XGBoost) and reimplementing the algorithms from basic principles. Compare the results, performance, and understanding of the algorithms from these two approaches.",
      "Master the classification problem-solving process: Practice skills through the steps of a typical machine learning project, including: loading and exploring data, data preprocessing (normalization, handling class imbalance), model training, evaluating results using appropriate metrics (Accuracy, F1-score).",
      "Analyze and discuss results: Compare the performance between different algorithms as well as between the library versions and the from-scratch implementations. From there, draw conclusions about the advantages and disadvantages of each model and implementation method in the context of the heartbeat classification problem."
    ]
  },
  theoretical: {
    preprocessing: {
      scaling: "Data normalization brings features to the same scale, which is necessary for many algorithms. Z-score normalization is a common method, transforming data according to the formula zi = (xi - μ) / σ. In this project, StandardScaler from scikit-learn is used.",
      imbalance: "Imbalanced data (where the minority class has fewer samples than the majority class) causes the model to be biased towards the majority class. Oversampling techniques like SMOTE (Synthetic Minority Over-sampling Technique) are used."
    },
    algorithms: {
      svm: "SVM classifies by finding the optimal separating hyperplane. For non-linear ECG data, the RBF kernel is used to map data to a higher-dimensional space. The regularization parameter C controls the trade-off between a large margin and classification error. Multi-class SVM uses One-vs-Rest or One-vs-One strategies. The project implements SVM with RBF kernel using the SVC library and also from scratch via Quadratic Programming.",
      logistic: "Logistic Regression is a linear model for classification, using a link function (Sigmoid for binary, Softmax for multi-class) to estimate probabilities. For the multi-class heartbeat problem, Softmax Regression is applied. The model is trained by minimizing the loss function (Negative Log-Likelihood).",
      gradient: "Gradient Boosting is an ensemble technique that sequentially builds models from weak learners (decision trees), correcting errors based on gradients. XGBoost (Extreme Gradient Boosting) is an optimized implementation of Gradient Boosting, outstanding in speed and performance due to improvements like regularization, handling missing values, and parallel computing. The project uses XGBoost for heartbeat classification, leveraging the power of boosting techniques."
    },
    metrics: "Evaluating the classification model on the test data uses appropriate metrics, especially for imbalanced data. The two main metrics are: Accuracy - The ratio of correct predictions to the total number of samples. Easy to understand but can be misleading with imbalanced data (Accuracy = Number of correct predictions / Total number of samples); F1-score - The harmonic mean of Precision and Recall, suitable for imbalanced data (F1-score = 2 × (Precision × Recall) / (Precision + Recall)). For multi-class problems, Macro F1-score (average F1 of each class) and Weighted F1-score (average F1 weighted by the number of samples) are used. The project uses both Accuracy and F1-score (macro and weighted) for a comprehensive evaluation."
  },
  methods: {
    preprocessing: "This section describes the dataset and the steps to prepare the data for model training. The heartbeat dataset is taken from the file mitbih_train.csv, containing 87,553 samples and 188 attributes. All attributes are of type float64. The initial data is loaded into a pandas DataFrame. Checking for missing values is performed (none detected in this file), then the data is split into features X and labels y. X has shape (87553, 187), and y has shape (87553,). Data normalization is applied to X using StandardScaler (Z-score normalization) so that the features have mean 0 and variance 1. The data is split into training (80%) and test (20%) sets using train_test_split (test_size=0.2, random_state=42). To handle class imbalance, the Oversampling technique with SMOTE is applied to the training set. SMOTE generates synthetic samples for the minority class, helping to balance the class distribution (for example, the training set after SMOTE has 289,465 samples). The test set retains the original distribution.",
    implementation: {
      libraries: {
        gradient: "The Gradient Boosting Classifier model is implemented using the GradientBoostingClassifier class from the scikit-learn library (sklearn.ensemble). The configuration parameters include the number of estimators (n_estimators=3), learning rate (learning_rate=0.05), maximum depth of trees (max_depth=5), and the training dataset after handling imbalance (X_train_resampled, y_train_resampled).",
        xgboost: "The XGBoost model is implemented using the xgboost library. The training and test data are converted to the optimized DMatrix format for XGBoost. The model is trained using the xgb.train function with specific parameters for the multi-class classification problem ('objective': 'multi:softmax', 'num_class': 5). Other parameters such as learning rate ('learning_rate': 0.1) and 'colsample_bytree': 0.8 are also configured. The model is trained on the training dataset after handling imbalance.",
        svm: "The SVM model is implemented using the SVC class from the scikit-learn library (sklearn.svm) with the RBF kernel (kernel='rbf', gamma='scale'). The model is trained on the training dataset after handling imbalance (X_train_resampled, y_train_resampled).",
        logistic: "The Logistic Regression model is implemented using the LogisticRegression class from the scikit-learn library (sklearn.linear_model). The configuration for the problem (penalty='l1', C=0.01) is applied. This model is trained on the original training dataset (before handling imbalance), unlike the other models."
      },
      scratch: {
        svm: "The from-scratch implementation of SVM focuses on binary classification using the RBF kernel, based on solving the Quadratic Programming (QP) problem for the dual form of SVM. A custom class named SVMQP is built to encapsulate this logic. Detailed source code is available in cs114_heartbeattest-3.ipynb. The process includes initializing parameters (epsilon, C, kernel, gamma), computing the kernel matrix, solving the QP problem with cvxopt, identifying support vectors, calculating the bias, and making predictions. For multi-class, it’s used within scikit-learn’s OneVsRestClassifier.",
        logistic: "The from-scratch implementation of Logistic Regression focuses on the multi-class case (Softmax Regression) in the MyLogisticRegression class, based on optimizing the Negative Log-Likelihood with gradient methods. Detailed source code is in cs114_heartbeattest-3.ipynb. It involves one-hot encoding the target, computing the loss and gradient, optimizing with scipy.optimize.minimize, and predicting probabilities and classes."
      }
    },
    environment: "The experiments were conducted on Kaggle Notebooks, providing computational resources and libraries for machine learning tasks. Configuration includes: Platform - Kaggle Notebooks; Hardware - GPU for training acceleration, especially XGBoost; OS - Linux; Python - 3.10; Libraries - numpy, pandas, sklearn, imblearn (SMOTE), cvxopt, scipy, xgboost. GPU usage significantly reduces training time for complex models and large datasets post-oversampling."
  },
  results: {
    library: {
      text: "This section presents the performance evaluation results of the classification models implemented using popular machine learning libraries on the test dataset. The metrics used are Accuracy, Precision, Recall and F1-score, as explained in Section 2.3. Below is a sample evaluation (replace with actual results from the report):",
      table: [
        { model: "Gradient Boosting", accuracy: "0.74", macroF1: "0.53", macroPrecision: "0.50", macroRecall: "0.69" },
        { model: "XGBoost", accuracy: "0.90", macroF1: "0.78", macroPrecision: "0.74", macroRecall: "0.90" },
        { model: "SVM", accuracy: "0.94", macroF1: "0.77", macroPrecision: "0.72", macroRecall: "0.91"  },
        { model: "Logistic Regression", accuracy: "0.67", macroF1: "0.48", macroPrecision: "0.45", macroRecall: "0.77" }
      ]
    }
  }
};

import { motion, useInView } from 'framer-motion';

const HowItWorksPage = () => {
  // Create refs for each section to track visibility
  const introductionRef = useRef(null);
  const theoreticalRef = useRef(null);
  const methodsRef = useRef(null);
  const resultsRef = useRef(null);

  // Use useInView to detect when each section is in view
  const isIntroductionInView = useInView(introductionRef, { once: false, margin: '-100px' });
  const isTheoreticalInView = useInView(theoreticalRef, { once: false, margin: '-100px' });
  const isMethodsInView = useInView(methodsRef, { once: false, margin: '-100px' });
  const isResultsInView = useInView(resultsRef, { once: false, margin: '-100px' });

  return (
    <div className="bg-[#363636] text-gray-200 min-h-screen grow w-full">
      <header className="bg-[#363636] p-6 text-center">
        <h1 className="text-4xl font-bold text-[#FFD700]">How It Works</h1>
      </header>
      <nav className="sticky top-0 bg-[#2D2D2D] p-4 shadow">
        <ul className="flex justify-center space-x-6">
          <li><a href="#introduction" className="text-[#FFD700] font-medium hover:text-white transition-colors">Introduction</a></li>
          <li><a href="#theoretical" className="text-[#FFD700] font-medium hover:text-white transition-colors">Theoretical Background</a></li>
          <li><a href="#methods" className="text-[#FFD700] font-medium hover:text-white transition-colors">Experimental Methods</a></li>
          <li><a href="#results" className="text-[#FFD700] font-medium hover:text-white transition-colors">Results</a></li>
        </ul>
      </nav>
      <main>
        <motion.section
          ref={introductionRef}
          initial={{ opacity: 0, y: 50 }}
          animate={isIntroductionInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 50 }}
          transition={{ duration: 0.5 }}
          id="introduction"
          className="max-w-11/12 mx-auto my-8 p-4 bg-[#404040] rounded-lg shadow border-l-4 border-[#FFD700]"
        >
          <h2 className="text-3xl font-semibold text-[#FFD700]">Introduction</h2>
          <div className="w-12 h-1 bg-[#FFD700] mt-2"></div>
          <h3 className="text-xl font-medium mt-6 mb-3 text-[#FFD700]">Problem Introduction</h3>
          <p className="leading-relaxed">{reportContent.introduction.problem}</p>
          <h3 className="text-xl font-medium mt-6 mb-3 text-[#FFD700]">Dataset Introduction</h3>
          <p className="leading-relaxed">{reportContent.introduction.dataset}</p>
          <h3 className="text-xl font-medium mt-6 mb-3 text-[#FFD700]">Project Objectives</h3>
          <ul className="list-disc pl-5 leading-relaxed">
            {reportContent.introduction.objectives.map((obj, index) => (
              <li key={index}>{obj}</li>
            ))}
          </ul>
        </motion.section>
        <motion.section
          ref={theoreticalRef}
          initial={{ opacity: 0, y: 50 }}
          animate={isTheoreticalInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 50 }}
          transition={{ duration: 0.5 }}
          id="theoretical"
          className="max-w-11/12 mx-auto my-8 p-4 bg-[#404040] rounded-lg shadow border-l-4 border-[#FFD700]"
        >
          <h2 className="text-3xl font-semibold text-[#FFD700]">Theoretical Background</h2>
          <div className="w-12 h-1 bg-[#FFD700] mt-2"></div>
          <h3 className="text-xl font-medium mt-6 mb-3 text-[#FFD700]">Data Preprocessing</h3>
          <p className="leading-relaxed"><strong>Data Scaling:</strong> {reportContent.theoretical.preprocessing.scaling}</p>
          <p className="leading-relaxed"><strong>Handling Imbalanced Data:</strong> {reportContent.theoretical.preprocessing.imbalance}</p>
          <h3 className="text-xl font-medium mt-6 mb-3 text-[#FFD700]">Classification Algorithms</h3>
          <p className="leading-relaxed"><strong>Support Vector Machine (SVM):</strong> {reportContent.theoretical.algorithms.svm}</p>
          <p className="leading-relaxed"><strong>Logistic Regression:</strong> {reportContent.theoretical.algorithms.logistic}</p>
          <p className="leading-relaxed"><strong>Gradient Boosting and XGBoost:</strong> {reportContent.theoretical.algorithms.gradient}</p>
          <h3 className="text-xl font-medium mt-6 mb-3 text-[#FFD700]">Evaluation Metrics</h3>
          <p className="leading-relaxed">{reportContent.theoretical.metrics}</p>
        </motion.section>
        <motion.section
          ref={methodsRef}
          initial={{ opacity: 0, y: 50 }}
          animate={isMethodsInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 50 }}
          transition={{ duration: 0.5 }}
          id="methods"
          className="max-w-11/12 mx-auto my-8 p-4 bg-[#404040] rounded-lg shadow border-l-4 border-[#FFD700]"
        >
          <h2 className="text-3xl font-semibold text-[#FFD700]">Experimental Methods</h2>
          <div className="w-12 h-1 bg-[#FFD700] mt-2"></div>
          <h3 className="text-xl font-medium mt-6 mb-3 text-[#FFD700]">Data Description and Preprocessing</h3>
          <p className="leading-relaxed">{reportContent.methods.preprocessing}</p>
          <h3 className="text-xl font-medium mt-6 mb-3 text-[#FFD700]">Model Implementation</h3>
          <p className="leading-relaxed"><strong>Using Libraries:</strong></p>
          <p className="leading-relaxed"><em>Gradient Boosting:</em> {reportContent.methods.implementation.libraries.gradient}</p>
          <p className="leading-relaxed"><em>XGBoost:</em> {reportContent.methods.implementation.libraries.xgboost}</p>
          <p className="leading-relaxed"><em>SVM:</em> {reportContent.methods.implementation.libraries.svm}</p>
          <p className="leading-relaxed"><em>Logistic Regression:</em> {reportContent.methods.implementation.libraries.logistic}</p>
          <p className="leading-relaxed"><strong>From-Scratch Implementation:</strong></p>
          <p className="leading-relaxed"><em>SVM:</em> {reportContent.methods.implementation.scratch.svm}</p>
          <p className="leading-relaxed"><em>Logistic Regression:</em> {reportContent.methods.implementation.scratch.logistic}</p>
          <h3 className="text-xl font-medium mt-6 mb-3 text-[#FFD700]">Experimental Environment</h3>
          <p className="leading-relaxed">{reportContent.methods.environment}</p>
        </motion.section>
        <motion.section
          ref={resultsRef}
          initial={{ opacity: 0, y: 50 }}
          animate={isResultsInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 50 }}
          transition={{ duration: 0.5 }}
          id="results"
          className="max-w-11/12 mx-auto my-8 p-4 bg-[#404040] rounded-lg shadow border-l-4 border-[#FFD700]"
        >
          <h2 className="text-3xl font-semibold text-[#FFD700]">Results and Discussion</h2>
          <div className="w-12 h-1 bg-[#FFD700] mt-2"></div>
          <h3 className="text-xl font-medium mt-6 mb-3 text-[#FFD700]">Evaluation Results of Library Models</h3>
          <p className="leading-relaxed">{reportContent.results.library.text}</p>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-600 my-4">
              <thead className="bg-[#2D2D2D]">
                <tr>
                  <th className="p-3 text-left font-semibold text-gray-200">Model</th>
                  <th className="p-3 text-left font-semibold text-gray-200">Accuracy</th>
                  <th className="p-3 text-left font-semibold text-gray-200">Macro Precision</th>
                  <th className="p-3 text-left font-semibold text-gray-200">Macro Recall</th>
                  <th className="p-3 text-left font-semibold text-gray-200">Macro F1-score</th>
                </tr>
              </thead>
              <tbody className="bg-[#363636]">
                {(() => {
                  // Find maximum values for each metric
                  const maxAccuracy = Math.max(...reportContent.results.library.table.map(row => parseFloat(row.accuracy) || 0));
                  const maxMacroPrecision = Math.max(...reportContent.results.library.table.map(row => parseFloat(row.macroPrecision) || 0));
                  const maxMacroF1 = Math.max(...reportContent.results.library.table.map(row => parseFloat(row.macroF1) || 0));
                  const maxMacroRecall = Math.max(...reportContent.results.library.table.map(row => parseFloat(row.macroRecall) || 0));

                  return reportContent.results.library.table.map((row, index) => (
                    <tr key={index} className="border-b border-gray-600">
                      <td className="p-3 text-gray-200">{row.model}</td>
                      <td className={`p-3 text-gray-200 ${parseFloat(row.accuracy) === maxAccuracy ? 'font-bold' : ''}`}>
                        {row.accuracy}
                      </td>
                      <td className={`p-3 text-gray-200 ${parseFloat(row.macroPrecision) === maxMacroPrecision ? 'font-bold' : ''}`}>
                        {row.macroPrecision}
                      </td>
                      <td className={`p-3 text-gray-200 ${parseFloat(row.macroRecall) === maxMacroRecall ? 'font-bold' : ''}`}>
                        {row.macroRecall}
                      </td>
                      <td className={`p-3 text-gray-200 ${parseFloat(row.macroF1) === maxMacroF1 ? 'font-bold' : ''}`}>
                        {row.macroF1}
                      </td>
                    </tr>
                  ));
                })()}
              </tbody>
            </table>
          </div>
        </motion.section>
      </main>
    </div>
  );
};


export default HowItWorksPage;