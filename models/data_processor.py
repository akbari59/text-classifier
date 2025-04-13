
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import joblib


# Preload tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

class DataPreprocessor:
    def __init__(self, file_path, text_col='description', mode='tfidf', label_col='label', vectorizer_params=None, random_state=42):
        """
        Parameters:
        - file_path (str): Path to the CSV file (e.g., 'data/trial.csv').
        - text_col (str): Name of the column containing raw trial descriptions.
        - label_col (str): Name of the column containing the disease labels.
        - vectorizer_params (dict): Parameters for the TfidfVectorizer.
        - random_state (int): Random seed for reproducibility.
        """
        self.file_path = file_path
        self.text_col = text_col
        self.label_col = label_col
        self.mode = mode
        self.random_state = random_state
        self.vectorizer_params = vectorizer_params or {"stop_words": "english", "max_features": 10000}
        self.vectorizer = TfidfVectorizer(**self.vectorizer_params)
        self.encoder = OneHotEncoder()

        self.df = None

    # Sample medical abbreviation dictionary
    ABBREVIATION_MAP = {
        'OCD': 'Obsessive Compulsive Disorder',
        'ALS': 'Amyotrophic Lateral Sclerosis',
        'PD': 'Parkinsons Disease',
        'COPD': 'Chronic Obstructive Pulmonary Disease',
        'AD': 'Alzheimers Disease',
        # Add more based on UMLS/SNOMED if available
    }

    def expand_abbreviation(self, word):
        """Expand a word using ABBREVIATION_MAP if it exists."""
        return self.ABBREVIATION_MAP.get(word.upper(), word)

    def clean_text(self, text, mode='smart'):
        """
        Preprocess clinical trial text with optional abbreviation expansion.

        Modes:
        - 'normal': standard lowercase cleanup
        - 'smart': preserve all-uppercase words
        - 'expand': expand medical abbreviations using ABBREVIATION_MAP
        """
        words = text.split()
        processed = []

        for word in words:
            if mode == 'smart':
                if word.isupper():
                    processed.append(word)
                else:
                    processed.append(word.lower())
            elif mode == 'expand':
                expanded = self.expand_abbreviation(word)
                processed.append(expanded.lower())
            else:  # 'normal'
                processed.append(word.lower())

        text = ' '.join(processed)
        text = re.sub(r'\d+', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))

        tokens = word_tokenize(text)
        tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]

        return ' '.join(tokens)

    def load_data(self):
        """Loads data from the CSV file and creates a 'clean_text' column if it doesn't exist."""
        self.df = pd.read_csv(self.file_path)
        self.df['clean_text'] = self.df[self.text_col].astype(str).apply(self.clean_text)
        self.data = self.df['clean_text']
        self.labels = self.df[self.label_col]
        return self.df

    def split_data(self, test_size=0.3, val_size=0.5):
        """
        Splits data into training, validation, and test sets.
        First splits training and temporary sets, then splits temporary set equally into validation and test.
        """
        data_train, data_temp, label_train, label_temp = train_test_split(
            self.data, self.labels, test_size=test_size, stratify=self.labels, random_state=self.random_state
        )
        data_val, data_test, label_val, label_test = train_test_split(
            data_temp, label_temp, test_size=val_size, stratify=label_temp, random_state=self.random_state
        )
        self.data_train, self.data_val, self.data_test = data_train, data_val, data_test
        self.label_train, self.label_val, self.label_test = label_train, label_val, label_test
        return (data_train, label_train), (data_val, label_val), (data_test, label_test)

    def vectorize_data(self):
        """Fits the TF‑IDF vectorizer on training data and transforms train, validation, and test sets."""
        # TODO! can use onehot encoder from sklearn
        self.X_train = self.vectorizer.fit_transform(self.data_train)
        self.y_train = self.label_train
        self.X_val = self.vectorizer.transform(self.data_val)
        self.y_val = self.label_val
        self.X_test = self.vectorizer.transform(self.data_test)
        self.y_test = self.label_test

        return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test

    def transformer_features(self, transformer_model='emilyalsentzer/Bio_ClinicalBERT'):
        from transformer_utils import TransformerFeatureExtractor
        if transformer_model is None:
            transformer_model = 'emilyalsentzer/Bio_ClinicalBERT'
        extractor = TransformerFeatureExtractor(model_name=transformer_model)

        self.X_train = extractor.encode(self.data_train.tolist())
        self.X_val = extractor.encode(self.data_val.tolist())
        self.X_test = extractor.encode(self.data_test.tolist())

        self.y_train = self.label_train
        self.y_val = self.label_val
        self.y_test = self.label_test

    def run_preprocessing(self, use_transformer=False, transformer_model=None):
        """Executes the full preprocessing pipeline and returns a dictionary of preprocessed data."""
        self.load_data()
        self.split_data()
        if use_transformer:
            self.transformer_features(transformer_model)
        else:
            self.vectorize_data()
        return {
            "X_train": self.X_train,
            "y_train": self.y_train,
            "X_val": self.X_val,
            "y_val": self.y_val,
            "X_test": self.X_test,
            "y_test": self.y_test
        }

    def save_vectorizer(self, path='vectorizer.pkl'):
        """Saves the trained TF-IDF vectorizer to disk."""
        joblib.dump(self.vectorizer, path)

    def load_vectorizer(self, path='vectorizer.pkl'):
        """Loads a saved TF-IDF vectorizer from disk."""
        self.vectorizer = joblib.load(path)

    def extract_features(self, text):
        """
        Extract features using the loaded vectorizer (TF-IDF only).
        """
        if not hasattr(self, 'vectorizer') or self.vectorizer is None:
            raise ValueError("Vectorizer is not loaded.")
        return self.vectorizer.transform([text])