import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_distances
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense
import joblib

import psycopg2 as pg2
from psycopg2.extras import RealDictCursor
import time
import pandas as pd

from dotenv import load_dotenv
import os

class RecommenderSystem:
    def __init__(self, data:pd.core.frame.DataFrame, content_col, encoder_path = None, model_path = None):
        if 'id' not in data.columns:
            raise ValueError("The DataFrame must contain 'id' columns.")
        if content_col is None:
            raise ValueError("You need to spesify content_col and ensure the DataFrame contains it.")
        self.df = data
        self.content_col = content_col
        if encoder_path is not None:
            self.encoder = joblib.load(encoder_path)
        else: #to retrain model to get new encoder
            self.encoder = None
        if model_path is not None:
            self.model = load_model(model_path)
        else: #to retrain model to get new model
            self.model = None

    def preprocess_data(self):
        X_train, X_val, y_train, y_val = train_test_split(
            self.df['metadata'], self.df['field'], test_size=0.2, random_state=42,# stratify=self.df['field']
        )

        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_val_encoded = label_encoder.transform(y_val)

        return X_train, X_val, y_train_encoded, y_val_encoded

    def train_model(self, X_train, y_train, X_val, y_val, epochs=10, save_model = None, save_encoder = None):
        self.encoder = CountVectorizer(stop_words="english", tokenizer=word_tokenize)
        X_train_encoded = self.encoder.fit_transform(X_train).toarray()
        X_val_encoded = self.encoder.transform(X_val).toarray()

        model = Sequential([
            Dense(128, activation='relu', input_shape=(X_train_encoded.shape[1],)),
            Dense(64, activation='relu'),
            Dense(len(np.unique(y_train)), activation='softmax')
        ])

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        history = model.fit(X_train_encoded, y_train, epochs=epochs, validation_data=(X_val_encoded, y_val), verbose=1)

        # Extract final training and validation accuracy
        final_train_accuracy = history.history['accuracy'][-1]
        final_val_accuracy = history.history['val_accuracy'][-1]

        print(f"Final Training Accuracy: {final_train_accuracy * 100:.2f}%")
        print(f"Final Validation Accuracy: {final_val_accuracy * 100:.2f}%")

        # Save the trained model
        if save_model is not None:
            model.save(save_model)
        # Save the labeled encoder
        if save_encoder is not None:
            joblib.dump(self.encoder, save_encoder)
        return model, history

    def plot_accuracy(self, history):
        # Plot training and validation accuracy
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

    def fit(self):
        self.encoder = CountVectorizer(stop_words="english", tokenizer=word_tokenize)
        self.bank = self.encoder.fit_transform(self.df[self.content_col])

    def recommend(self, idx:str, topk=10):
        content = self.df.loc[self.df['id']==idx, self.content_col].values[0]
        code = self.encoder.transform([content])
        dist = cosine_distances(code, self.bank)
        rec_idx = dist.argsort()[0, 1:(topk+1)]
        return self.df.loc[rec_idx,'id'].values

    def run(self,
            graph_history=False,
            save_model=None,
            save_encoder=None):
        X_train, X_val, y_train, y_val = self.preprocess_data()
        self.model, history = self.train_model(X_train, y_train, X_val, y_val, epochs=10,save_model=save_model,save_encoder=save_encoder)
        if graph_history == True:
            self.plot_accuracy(history)

class Load_Data_CB():
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
    
    def load_data(self,table=['sponsorship','rentals','media_partner']):
        self.cur.execute(f'''
        SELECT * FROM {table}
        ''')
        array = self.cur.fetchall()
        DF_PROCESS = pd.DataFrame(array)
        DF_PROCESS['description'] = DF_PROCESS['description'].str.replace('\n', ' ')
        DF_PROCESS['metadata'] = 'Sponsor'+" " + DF_PROCESS['field'] +" "+ DF_PROCESS['description'] 
        return DF_PROCESS
    
    def close_database(self):
        if self.cur:
            self.cur.close()
        if self.conn:
            self.conn.close()
        print('The database has been closed.')
        return None
    
# lo = Load_Data_CB()
# df_lo = lo.load_data('media_partner')
# recsys_model = RecommenderSystem(df_lo, content_col="metadata",
#                                  encoder_path='media_partner_encoder.joblib',
#                                  model_path='media_partner_cb_model.h5')
# # recsys_model.run(save_encoder='media_partner_encoder.joblib',save_model='media_partner_cb_model.h5')
# recsys_model.fit()
# print(recsys_model.recommend('6b0b5bf3-16f4-4405-b0ae-4313228ee31f'))
# print(recsys_model.recommend('c2a2b772-6696-476e-b230-7eda70a1d453'))