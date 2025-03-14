import os
import numpy as np
import joblib
import tensorflow as tf
import streamlit as st
import nltk
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


nltk.download("punkt")
nltk.download("stopwords")

MODEL_PATH = "artifacts/model_trainer/model.keras"
SCALER_PATH = "artifacts/model_trainer/scaler.pkl"
WORD2VEC_PATH = "artifacts/data_transformation/word2vec.model"


model = tf.keras.models.load_model(MODEL_PATH)


scaler = joblib.load(SCALER_PATH)

word2vec_model = Word2Vec.load(WORD2VEC_PATH)
VECTOR_SIZE = 100  


CONFIDENCE_THRESHOLD = 0.6  


def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())  
    words = [word for word in words if word.isalnum() and word not in stop_words]
    return words


def text_to_vector(text, word2vec, scaler):
    words = preprocess_text(text)
    word_vectors = [word2vec.wv[word] for word in words if word in word2vec.wv]
    
    if len(word_vectors) == 0:
        return np.zeros((1, word2vec.vector_size))  

    sentence_vector = np.mean(word_vectors, axis=0).reshape(1, -1)
    return scaler.transform(sentence_vector)  


st.title(" SQL Injection Detection")



user_input = st.text_area("Enter SQL query:", height=150)

if st.button("Analyze Query"):
    if user_input.strip():
        
        input_vector = text_to_vector(user_input, word2vec_model, scaler)
        input_vector = input_vector.reshape((1, 1, input_vector.shape[1]))  

        
        prediction = model.predict(input_vector)[0][0]
        is_sql_injection = prediction > CONFIDENCE_THRESHOLD

        
        st.subheader("Prediction Result:")
        if is_sql_injection:
            st.error(f" **SQL Injection Detected!** (Confidence: {prediction:.2f})")
        else:
            st.success(f" **Query is Safe.** ")
    else:
        st.warning(" Please enter a query to analyze.")