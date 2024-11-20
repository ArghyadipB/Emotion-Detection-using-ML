from src.ML_emotion_detection import logger
from sklearn.model_selection import train_test_split
import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from src.ML_emotion_detection.utils.common import save_bin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from src.ML_emotion_detection.entity.config_entity import DataTransformationConfig
import os
tqdm.pandas()


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.tfidf = TfidfVectorizer(max_features=4000)

    def train_test_spliting(self, test_size=0.2):
       
        data = pd.read_parquet('E:/Projects/E2E Emotion Detection from text/Emotion-Detection-using-ML/artifacts/data_ingestion/train-00000-of-00001.parquet')

        logger.info("Split data into training and test sets")
        X_train, X_test, y_train, y_test = train_test_split(data['text'],
                                                            data['label'],
                                                            test_size=test_size,
                                                            stratify=data['label'],
                                                            random_state=42)

        save_bin(y_train, os.path.join(self.config.root_dir, "y_train.joblib"))

        save_bin(y_test, os.path.join(self.config.root_dir, "y_test.joblib"))

        return X_train, X_test

    def preprocess(self, text, *args):
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'#\w+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)

        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words]

        return ' '.join(tokens)

    def pos_count_features(self, text, *args):
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

    def text_feature_extraction(self):


        text_column = 'text'
        numerical_columns = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN',
              'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X']
        label_columns = 'label'

        preprocessor = ColumnTransformer(
            transformers=[
                ('text', self.tfidf, text_column),  
                ('num', StandardScaler(), numerical_columns) 
            ],
        )

        return preprocessor

    def transform_and_save(self, preprocessor, X_train, X_test):
        
        X_train_processed = preprocessor.fit_transform(X_train)

        
        X_test_processed = preprocessor.transform(X_test)

        
        save_bin(X_train_processed, os.path.join(self.config.root_dir, "X_train.joblib"))

        
        save_bin(X_test_processed, os.path.join(self.config.root_dir, "X_test.joblib"))
        
        logger.info(f"Training set shape after preprocessing: {X_train_processed.shape}")
        logger.info(f"Test set shape after preprocessing: {X_test_processed.shape}")