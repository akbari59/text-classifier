
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import joblib


from models.data_processor import DataPreprocessor


class ModelTrainer:
    def __init__(self, model_type='naive_bayes', model_params=None, random_state=42):
        """
        Parameters:
        - model_type (str): Type of model to use ('naive_bayes', 'logistic_regression', 'svm').
        - model_params (dict): Additional parameters for the classifier.
        - random_state (int): Random seed for reproducibility.
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model_params = model_params or {}
        self.model = None

    def build_model(self):
        """Initializes the classifier based on the specified model type."""
        if self.model_type == 'naive_bayes':
            self.model = MultinomialNB(**self.model_params)
        elif self.model_type == 'logistic_regression':
            self.model = LogisticRegression(max_iter=1000, random_state=self.random_state, **self.model_params)
        elif self.model_type == 'svm':
            self.model = SVC(kernel='linear', random_state=self.random_state, **self.model_params)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        return self.model

    def fit(self, data_processor):
        """Trains the classifier on the vectorized training data."""
        """
        Runs the full training and evaluation pipeline using the preprocessed data.
        Evaluates the model on the validation set.
        """
        if self.model is None:
            self.build_model()
        self.model.fit(data_processor.X_train,  data_processor.y_train)
        return self.model

    def evaluate(self, X, y_true, title = ''):
        """
        Evaluates the model, printing a confusion matrix, classification report,
        and up to five misclassified examples.
        """
        y_pred = self.model.predict(X)

        # Plot confusion matrix
        labels = np.unique(y_true)
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title(f'Confusion Matrix: {title}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()

        # Print classification report
        print(f"Classification Report: {title}\n", classification_report(y_true, y_pred))

    def save_model(self, path='model.pkl'):
        """Saves the trained model to disk."""
        joblib.dump(self.model, path)

    def load_model(self, path='model.pkl'):
        """Loads a trained model from disk."""
        self.model = joblib.load(path)

    def predict(self, X):
        """Make predictions with the loaded model."""
        return self.model.predict(X)


# ----------------------------
# Example Usage
# ----------------------------
if __name__ == '__main__':
    # Step 1: Data Preprocessing
    data_preprocessor = DataPreprocessor(file_path='../data/trials.csv', text_col='description', label_col='label')
    preprocessed_data = data_preprocessor.run_preprocessing(use_transformer=False)


    # To try a different model, simply change the model_type:
    trainer_lr = ModelTrainer(model_type='logistic_regression')
    trainer_lr.fit(data_preprocessor)
    trainer_lr.save_model('model.pkl')
    data_preprocessor.save_vectorizer('vectorizer.pkl')


    trainer_lr.evaluate(data_preprocessor.X_train, data_preprocessor.y_train, title='Training')
    trainer_lr.evaluate(data_preprocessor.X_test, data_preprocessor.y_test, title='Testing')


