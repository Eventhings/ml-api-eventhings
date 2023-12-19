import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer,tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import joblib

import psycopg2 as pg2
from psycopg2.extras import RealDictCursor
import time
import pandas as pd

from dotenv import load_dotenv
import os


class NLP_emotion:
    def __init__(self,
                 model_path,
                 label_encoder_path,
                 tokenizer_path, 
                 set_of_sentences:np.ndarray):
        model =  tf.keras.models.load_model(model_path)
        label_encoder = joblib.load(label_encoder_path)
        with open(tokenizer_path, "r", encoding="utf-8") as json_file:
            loaded_tokenizer_json = json_file.read()
        tokenizer = tokenizer_from_json(loaded_tokenizer_json)
        self.model=model
        self.label_encoder=label_encoder
        self.tokenizer=tokenizer
        self.sentences=set_of_sentences

    def clean_text(self,text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        return text

    def predict_emotions(self):
        max_length = 200
        cleaned_sentences = [self.clean_text(sentence) for sentence in self.sentences]  
        encoded_sentences = self.tokenizer.texts_to_sequences(cleaned_sentences)
        padded_sentences = pad_sequences(encoded_sentences, maxlen=max_length, padding='post')
        predictions = self.model.predict(padded_sentences)
        return predictions
    
    def percentage_emotions(self):
        predictions = self.predict_emotions()
        emotion_sums = np.sum(predictions, axis=0)
        emotion_averages = emotion_sums / len(predictions)
        portion_emotion = {label: float(portion) for label, portion in zip(self.label_encoder.classes_, emotion_averages)}
        return portion_emotion

class Load_Data_NLP():
    def __init__(self):
        dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
        load_dotenv(dotenv_path)
        DB_USERNAME = os.getenv('DB_USERNAME')
        DB_PASSWORD = os.getenv('DB_PASSWORD')
        DB_HOST = os.getenv('DB_HOST')
        DB_PORT = os.getenv('DB_PORT')
        DB_DATABASE = os.getenv('DB_DATABASE')
        DB_DATABASE_DEV = os.getenv('DB_DATABASE_DEV')
        while True:
            try:
                self.conn = pg2.connect(database = DB_DATABASE,
                                user= DB_USERNAME,
                                password=DB_PASSWORD, 
                                host=DB_HOST,
                                port=DB_PORT,
                                cursor_factory= RealDictCursor)
                self.cur = self.conn.cursor()
                print('Database connection was successful.')
                break
            except Exception as error:
                print("Connectiong to database failed.")
                print('Error: ', error)
                time.sleep(2)
        return None
    
    def load_data(self,table=['sponsorship_review','rentals_review','media_partner_review']):
        self.cur.execute(f'''
        SELECT * FROM {table}
        ''')
        array = self.cur.fetchall()
        DF_PROCESS = pd.DataFrame(array)
        return DF_PROCESS
    
    def close_database(self):
        if self.cur:
            self.cur.close()
        if self.conn:
            self.conn.close()
        print('The database has been closed.')
        return None
