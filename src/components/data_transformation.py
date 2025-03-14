import os
import pandas as pd
import nltk
import joblib
import numpy as np
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from src import logger
from src.entity.config_entity import DataTransformationConfig


nltk.download('punkt')
nltk.download('stopwords')

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def tokenize_and_remove_stopwords(self, sentence):
        """Tokenizes sentence and removes stopwords."""
        stop_words = set(stopwords.words('english'))
        words = word_tokenize(sentence.lower())  
        return [word for word in words if word.isalnum() and word not in stop_words]

    def sentence_to_vector(self, sentence, model, vector_size):
        """Converts a sentence to a numerical vector by averaging word embeddings."""
        words = self.tokenize_and_remove_stopwords(sentence)
        word_vectors = [model.wv[word] for word in words if word in model.wv]
        if len(word_vectors) == 0:
            return np.zeros(vector_size)  
        return np.mean(word_vectors, axis=0)  

    def transform_text(self):
        """Reads data, trains Word2Vec model, converts sentences to vectors, and saves train-test data."""
        
        data = pd.read_csv(self.config.data_path, encoding="utf-16")
        data.dropna(inplace=True)

        
        tokenized_sentences = data['Sentence'].astype(str).apply(self.tokenize_and_remove_stopwords)

        
        vector_size = 100  
        model = Word2Vec(sentences=tokenized_sentences, vector_size=vector_size, window=5, min_count=1, workers=4)

        
        word2vec_path = os.path.join(self.config.root_dir, "word2vec.model")
        model.save(word2vec_path)

        
        X = np.array([self.sentence_to_vector(sentence, model, vector_size) for sentence in data['Sentence']])
        y = data['Label'] 

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

     
        train_df = pd.DataFrame(X_train)
        train_df['Label'] = y_train.values
        train_df.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)

        test_df = pd.DataFrame(X_test)
        test_df['Label'] = y_test.values
        test_df.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)

        logger.info("Data transformation complete. Train, test datasets, and Word2Vec model saved.")
        print("Data transformation complete. Train, test datasets, and Word2Vec model saved.")
