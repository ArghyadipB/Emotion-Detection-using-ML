from src.ML_emotion_detection import logger
from sklearn.model_selection import train_test_split
import pandas as pd
import re
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from src.ML_emotion_detection.utils.common import save_bin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from src.ML_emotion_detection.entity.config_entity import \
    DataTransformationConfig
import os
tqdm.pandas()


class TextPreprocessor(BaseEstimator, TransformerMixin):
    """
    A custom text preprocessing transformer for text data.

    This class handles text cleaning, lowercasing, stop word removal,
    and lemmatization.

    Attributes:
        stop_words (set): Set of English stop words.
        lemmatizer (WordNetLemmatizer): Lemmatizer for reducing words to their
        base form.
    """

    def __init__(self):
        """
        Initializes the TextPreprocessor with stop words and lemmatizer.
        """
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def fit(self, X, y=None):
        """
        Fit method (does nothing as no fitting is required).

        Args:
            X (pd.Series): Input text data.
            y (pd.Series, optional): Target labels. Defaults to None.

        Returns:
            self: The fitted TextPreprocessor object.
        """
        return self

    def transform(self, X):
        """
        Applies text preprocessing to the input data.

        Args:
            X (pd.Series): Input text data.

        Returns:
            pd.DataFrame: Preprocessed text data with one column 'text'.
        """
        processed = X.progress_apply(self._preprocess)
        return pd.DataFrame(processed, columns=['text'])

    def _preprocess(self, text):
        """
        Preprocesses a single text input by cleaning, tokenizing, and
        lemmatizing.

        Args:
            text (str): Input text.

        Returns:
            str: Preprocessed text.
        """
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'#\w+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word
                  not in self.stop_words]
        return ' '.join(tokens)


class POSCountFeatures(BaseEstimator, TransformerMixin):
    """
    A custom transformer to count parts-of-speech (POS) features.

    Attributes:
        None
    """

    def fit(self, X, y=None):
        """
        Fit method (does nothing as no fitting is required).

        Args:
            X (pd.DataFrame): Input text data.
            y (pd.Series, optional): Target labels. Defaults to None.

        Returns:
            self: The fitted POSCountFeatures object.
        """
        return self

    def transform(self, X):
        """
        Applies POS counting to the input text data.

        Args:
            X (pd.DataFrame): Input text data.

        Returns:
            pd.DataFrame: Text data with additional POS count features.
        """
        pos_count = X['text'].progress_apply(self._get_pos_counts)
        return pd.concat([X, pos_count], axis=1)

    def _get_pos_counts(self, text):
        """
        Counts parts-of-speech (POS) in the input text.

        Args:
            text (str): Input text.

        Returns:
            pd.Series: Counts of different POS tags.
        """
        tokens = word_tokenize(text)
        tagged_tokens = pos_tag(tokens)

        pos_counts = {
            'ADJ': 0, 'ADP': 0, 'ADV': 0, 'AUX': 0, 'CCONJ': 0, 'DET': 0,
            'INTJ': 0, 'NOUN': 0, 'NUM': 0, 'PART': 0, 'PRON': 0, 'PROPN': 0,
            'PUNCT': 0, 'SCONJ': 0, 'SYM': 0, 'VERB': 0, 'X': 0
        }

        for token, tag in tagged_tokens:
            if tag.startswith('JJ'):
                pos_counts['ADJ'] += 1
            elif tag.startswith('RB'):
                pos_counts['ADV'] += 1
            elif tag.startswith('VB'):
                pos_counts['VERB'] += 1
            elif tag.startswith('NN'):
                pos_counts['NOUN'] += 1
            elif tag == 'IN':
                pos_counts['ADP'] += 1
            elif tag == 'DT':
                pos_counts['DET'] += 1
            elif tag == 'PRP' or tag == 'PRP$':
                pos_counts['PRON'] += 1
            elif tag == 'TO':
                pos_counts['PART'] += 1
            elif tag == 'PDT':
                pos_counts['DET'] += 1
            elif tag == 'CD':
                pos_counts['NUM'] += 1
            elif tag == 'CC':
                pos_counts['CCONJ'] += 1
            elif tag == 'RP':
                pos_counts['PART'] += 1
            elif tag == ',':
                pos_counts['PUNCT'] += 1
            elif tag == 'SYM':
                pos_counts['SYM'] += 1
            elif tag == 'EX':
                pos_counts['X'] += 1
            else:
                pos_counts['X'] += 1

        return pd.Series(pos_counts)


class TfidfFeature:
    """
    A feature transformer that combines TF-IDF features and scaled numerical
    features.

    Attributes:
        preprocessor (ColumnTransformer): Transformer for text and numerical
        columns.
    """

    def __init__(self):
        """
        Initializes the TfidfFeature transformer with TF-IDF and numerical
        scaling.
        """
        text_column = 'text'
        numerical_columns = [
            'ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN',
            'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT',
            'SCONJ', 'SYM', 'VERB', 'X'
        ]

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('text', TfidfVectorizer(max_features=4000), text_column),
                ('num', StandardScaler(), numerical_columns)
            ]
        )

    def fit(self, X, y=None):
        """
        Fits the preprocessor to the input data.

        Args:
            X (pd.DataFrame): Input data with text and numerical features.
            y (pd.Series, optional): Target labels. Defaults to None.

        Returns:
            self: The fitted TfidfFeature object.
        """
        self.preprocessor.fit(X)
        return self

    def transform(self, X):
        """
        Transforms the input data using the fitted preprocessor.

        Args:
            X (pd.DataFrame): Input data with text and numerical features.

        Returns:
            np.ndarray: Transformed data.
        """
        return self.preprocessor.transform(X)

    def fit_transform(self, X, y=None):
        """
        Fits the preprocessor and transforms the input data.

        Args:
            X (pd.DataFrame): Input data with text and numerical features.
            y (pd.Series, optional): Target labels. Defaults to None.

        Returns:
            np.ndarray: Transformed data.
        """
        return self.preprocessor.fit_transform(X)


class DataTransformation:
    """
    A class for handling data transformation, including preprocessing,
    feature extraction, and train-test splitting.

    Attributes:
        config (DataTransformationConfig): Configuration object.
        stop_words (set): Set of English stop words.
        lemmatizer (WordNetLemmatizer): Lemmatizer for text preprocessing.
    """

    def __init__(self, config: DataTransformationConfig):
        """
        Initializes the DataTransformation object.

        Args:
            config (DataTransformationConfig): Configuration for data
            transformation.
        """
        self.config = config
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def train_test_spliting(self, test_size=0.2):
        """
        Splits data into training and testing sets.

        Args:
            test_size (float): Proportion of data to be used for testing.
            Defaults to 0.2.

        Returns:
            tuple: Training and testing
            data (X_train, X_test, y_train, y_test).
        """
        data = pd.read_parquet(
            'E:/Projects/E2E Emotion Detection from text/'
            'Emotion-Detection-using-ML/'
            'artifacts/data_ingestion/train-00000-of-00001.parquet'
        )

        logger.info("Split data into training and test sets")
        X_train, X_test, y_train, y_test = train_test_split(
            data['text'], data['label'], test_size=test_size,
            stratify=data['label'], random_state=42)

        save_bin(y_train, os.path.join(self.config.root_dir, "y_train.joblib"))
        save_bin(y_test, os.path.join(self.config.root_dir, "y_test.joblib"))

        return X_train, X_test

    def pipeline_and_transform(self, X_train, X_test):
        """
        Creates a preprocessing pipeline, transforms the data, and saves the
        outputs.

        Args:
            X_train (pd.Series): Training text data.
            X_test (pd.Series): Testing text data.

        Returns:
            tuple: Transformed training and testing
            data (X_train_processed, X_test_processed).
        """
        pipeline = Pipeline([
            ('text_preprocessor', TextPreprocessor()),
            ('pos_counter', POSCountFeatures()),
            ('tfidf_feature', TfidfFeature())
        ])

        logger.info("Transforming train and test data")
        X_train_processed = pipeline.fit_transform(X_train)
        X_test_processed = pipeline.transform(X_test)

        save_bin(
            pipeline,
            os.path.join(self.config.root_dir, "preprocessor.joblib")
        )
        save_bin(
            X_train_processed,
            os.path.join(self.config.root_dir, "X_train.joblib")
        )
        save_bin(
            X_test_processed,
            os.path.join(self.config.root_dir, "X_test.joblib")
        )

        logger.info(
            f"Training set shape after preprocessing:{X_train_processed.shape}"
        )

        logger.info(
            f"Test set shape after preprocessing:{X_test_processed.shape}"
        )
