import useEmblaCarousel from 'embla-carousel-react';
import Autoplay from 'embla-carousel-autoplay';
import CodeTypingEffect from '@/components/TypingCodeEffect';


// Code Snippets
const svmFromScratch = `import numpy as np

class SVM:
    def __init__(self, lr=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = lr
        self.lambda_param = lambda_param
        self.n_iters = n_iters

    def fit(self, X, y):
        y_ = np.where(y <= 0, -1, 1)
        self.w = np.zeros(X.shape[1])
        self.b = 0
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]

    def predict(self, X):
        return np.sign(np.dot(X, self.w) - self.b)`;

const logisticRegressionFromScratch = `import numpy as np

class LogisticRegression:
    def __init__(self, lr=0.01, num_iter=1000):
        self.lr = lr
        self.num_iter = num_iter

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        self.theta = np.zeros(X.shape[1])
        for _ in range(self.num_iter):
            z = np.dot(X, self.theta)
            h = self.sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.lr * gradient

    def predict(self, X):
        return self.sigmoid(np.dot(X, self.theta)) >= 0.5`;

const svmSklearn = `from sklearn.svm import SVC

model = SVC(kernel='linear')
model.fit(X_train, y_train)
predictions = model.predict(X_test)`;

const logisticRegressionSklearn = `from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)`;

const xgboostClassifier = `from xgboost import XGBClassifier

model = XGBClassifier()
model.fit(X_train, y_train)
predictions = model.predict(X_test)`;

// Main CodeShowcase Component
const CodeShowcase = () => {
  const [emblaRef, emblaApi] = useEmblaCarousel({ loop: true }, [Autoplay({ delay: 5000 })]);

  const scrollPrev = () => emblaApi && emblaApi.scrollPrev();
  const scrollNext = () => emblaApi && emblaApi.scrollNext();

  return (
    <div className="w-full min-w-300 max-w-4xl mx-auto py-12 min-h-screen flex flex-col items-center bg-[#0d0d0d] text-[#facc15]">
      <h1 className="text-4xl font-extrabold text-center mb-8 drop-shadow-md bg-gradient-to-r from-yellow-400 via-yellow-300 to-yellow-500 bg-clip-text text-transparent">
        Model Implementation
      </h1>
      <div className="embla w-full overflow-hidden rounded-lg shadow-lg border border-[#333]" ref={emblaRef}>
        <div className="min-h-60 embla__container flex">
          <div className="embla__slide flex-[0_0_100%] min-w-0 bg-[#1a1a1a] p-4 rounded-lg">
            <CodeTypingEffect title="SVM from Scratch" code={svmFromScratch} />
          </div>
          <div className="embla__slide flex-[0_0_100%] min-w-0 bg-[#1a1a1a] p-4 rounded-lg">
            <CodeTypingEffect title="Logistic Regression from Scratch" code={logisticRegressionFromScratch} />
          </div>
          <div className="embla__slide flex-[0_0_100%] min-w-0 bg-[#1a1a1a] p-4 rounded-lg">
            <CodeTypingEffect title="SVM (scikit-learn)" code={svmSklearn} />
          </div>
          <div className="embla__slide flex-[0_0_100%] min-w-0 bg-[#1a1a1a] p-4 rounded-lg">
            <CodeTypingEffect title="Logistic Regression (scikit-learn)" code={logisticRegressionSklearn} />
          </div>
          <div className="embla__slide flex-[0_0_100%] min-w-0 bg-[#1a1a1a] p-4 rounded-lg">
            <CodeTypingEffect title="XGBoost Classifier" code={xgboostClassifier} />
          </div>
        </div>
      </div>
      <div className="flex justify-center gap-4 mt-6">
        <button
          className="bg-[#facc15] hover:bg-yellow-400 text-black font-bold py-2 px-5 rounded-lg transition duration-300 shadow-md"
          onClick={scrollPrev}
        >
          Previous
        </button>
        <button
          className="bg-[#facc15] hover:bg-yellow-400 text-black font-bold py-2 px-5 rounded-lg transition duration-300 shadow-md"
          onClick={scrollNext}
        >
          Next
        </button>
      </div>
    </div>
  );
};
export default CodeShowcase;