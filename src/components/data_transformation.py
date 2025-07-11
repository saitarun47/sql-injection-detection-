import os
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import joblib
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re

from src import logger
from src.entity.config_entity import DataTransformationConfig

@dataclass
class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def convert_word_to_int(self, sentence, vocab):
        return [vocab.get(word, vocab['<unk>']) for word in sentence]

    def pad_features(self, reviews_int, seq_length):
        features = np.zeros((len(reviews_int), seq_length), dtype=int)
        for i, row in enumerate(reviews_int):
            if len(row) != 0:
                features[i, -len(row):] = np.array(row)[:seq_length]
        return features

    def transform_data(self):
        try:
            
            data = pd.read_csv(self.config.data_path, encoding="utf-16")
            logger.info(f"Loaded data with shape: {data.shape}")

            
            data['Sentence'] = data['Sentence'].astype(str)
            data = data.dropna(subset=['Sentence', 'Label'])

            
            initial_size = len(data)
            data = data.drop_duplicates(subset=['Sentence'])
            final_size = len(data)
            removed_duplicates = initial_size - final_size
            logger.info(f"Removed {removed_duplicates} duplicate samples")

            
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt')

            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                nltk.download('stopwords')

            
            def clean_text(text):
                if not isinstance(text, str):
                    text = str(text)
                
                text = text.lower()
                
                text = re.sub(r'[^a-zA-Z0-9\s<>=\-\+\*\/\(\)\[\]\{\}\.\,\;\:\'\"]', '', text)
                return text

            
            data['cleaned_text'] = data['Sentence'].apply(clean_text)
            logger.info("Text cleaning completed")

            # Tokenization
            data['tokenized'] = data['cleaned_text'].apply(word_tokenize)
            logger.info("Tokenization completed")

            
            all_words = []
            for tokens in data['tokenized']:
                all_words.extend(tokens)


            word_freq = {}
            for word in all_words:
                word_freq[word] = word_freq.get(word, 0) + 1

            vocab = {'<pad>': 0, '<unk>': 1}
            for word, freq in word_freq.items():
                if freq >= 2:
                    vocab[word] = len(vocab)

            logger.info(f"Vocabulary created with {len(vocab)} words")

            
            data['encoded'] = data['tokenized'].apply(lambda x: self.convert_word_to_int(x, vocab))

        
            train_temp, test_temp = train_test_split(
                data, 
                test_size=0.3, 
                random_state=42, 
                stratify=data['Label']
            )
        
            max_len = max(len(seq) for seq in train_temp['encoded'])
            logger.info(f"Max sequence length (from training data): {max_len}")

            # Pad sequences
            data['encoded_text'] = data['encoded'].apply(lambda x: self.pad_features([x], max_len)[0])

            
            train_data, test_data = train_test_split(
                data, 
                test_size=0.3, 
                random_state=42, 
                stratify=data['Label']
            )

            logger.info(f"Train set size: {len(train_data)}")
            logger.info(f"Test set size: {len(test_data)}")

           
            train_data.to_csv(self.config.train_data_path, index=False)
            test_data.to_csv(self.config.test_data_path, index=False)

            
            vocab_path = os.path.join(os.path.dirname(self.config.train_data_path), "vocab.pkl")
            params_path = os.path.join(os.path.dirname(self.config.train_data_path), "params.pkl")
            
            joblib.dump(vocab, vocab_path)
            
            params = {
                'vocab_size': len(vocab),
                'max_len': max_len,
                'pad_idx': vocab['<pad>']
            }
            joblib.dump(params, params_path)

            logger.info("Data transformation completed successfully")
            logger.info(f"Vocabulary saved to {vocab_path}")
            logger.info(f"Parameters saved to {params_path}")

        except Exception as e:
            logger.error(f"Error in data transformation: {e}")
            raise e
