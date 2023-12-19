import os
import pickle
import traceback
import numpy as np

from tensorflow.keras.models import load_model

from recsys.model.modelCF import *
from recsys.model.utils import *

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel

#------------------------------------------------------------------------------------------------------------------------------#

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
    # "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)


@app.get('/')
async def root():
    return {'message': 'Hello World'}


class Recommendation(BaseModel):
    user_id: str


@app.post('/recsys/cf/train')
def train():
    try:
        train_model_CF()

        dct = {
            'status': 200,
            'desc': 'model trained successfully.'
        }

    except:
        dct = {
            'status': 404,
            'desc': 'model training has failed.'
        }

    return dct

@app.post('/recsys/cf/recommend')
async def recommendItem(req: Recommendation):
    current_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'recsys')

    # Current user
    user_id = req.user_id

    # Load data
    uids, iids, df_train, df_test, df_neg, users, items, label_encoder_user, label_encoder_item = load_dataset()

    # Load CF model and its perforamnce
    modelCF_path = os.path.join(current_directory, 'model')
    modelCF = load_model(os.path.join(modelCF_path, 'modelCF.h5'))
    
    performance_path = os.path.join(current_directory, 'performance')
    with open(os.path.join(performance_path, 'hitrates_avg_CF.pkl'), 'rb') as f:
        hitrates_avg = pickle.load(f)
    with open(os.path.join(performance_path, 'ndcgs_avg_CF.pkl'), 'rb') as f:
        ndcgs_avg = pickle.load(f)


    try:
        # If user has not give any rating or has not click some vendor(s) within a session
        uids = label_encoder_user.inverse_transform(uids)
        if user_id in uids:
            # Predict ratings using CF model
            ratingsCF = predict_ratings_cf(
                user_id = user_id,
                items = items,
                model = modelCF,
                label_user = label_encoder_user,
                label_item = label_encoder_item
            )

            average_ratings, recommendation_items = get_top_k_items(ratingsCF, k = 10)
    
        # Else proceed to predict using CB model
        else:
            '''
            FILL THE CODE HERE FOR CB
            '''
            recommendation_items = []

        dct = {
            'status': 200,
            'message': 'recommendation for user has been successfully get.',
            'data': {
                'user_id': user_id,
                'recommendations': recommendation_items
                },
            'success': True,
            'error': None
        }

    except Exception as e:
        recommendation_items = []
        traceback.print_exc()

        dct = {
            'userID': user_id,
            'status': 404,
            'message': 'recommendation for user has failed.',
            'data': {
                'user_id': user_id,
                'recommendations': recommendation_items
                },
            'success': False,
            'error': str(e)
        }

    return dct