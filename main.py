import os
import pickle
import traceback
import numpy as np
from dotenv import dotenv_values
from tensorflow.keras.models import load_model

from recsys.model.model_cf import *
from recsys.model.utils import *

from fastapi import FastAPI,HTTPException,Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from pydantic import BaseModel

import nlp.Capstone_NLP_Emotions as cne
import importlib
importlib.reload(cne)

import recsys.model.Capstone_CB_Recsys as cb
import importlib
importlib.reload(cb)

load_dotenv()
#------------------------------------------------------------------------------------------------------------------------------#
app = FastAPI()
origins = [
    "*"
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
        train_model_cf()

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
    df_medpar, df_rental, df_sponsor = load_dataset()
    uids, iids, df_train, df_test, df_neg, users, items, label_encoder_user, label_encoder_item = transform_dataset(df_medpar, df_rental, df_sponsor)

    try:
        # If user has not give any rating or has not click some vendor(s) within a session
        uids = label_encoder_user.inverse_transform(uids)

        if user_id in uids:
            # Load CF model
            model_cf_path = os.path.join(current_directory, 'model')
            model_cf = load_model(os.path.join(model_cf_path, 'model_cf.h5'))

            # Predict ratings using CF model
            ratings_cf = predict_ratings_cf(
                user_id = user_id,
                items = items,
                model = model_cf,
                label_user = label_encoder_user,
                label_item = label_encoder_item
            )

            average_ratings, recommendation_items = get_top_k_items(ratings_cf, k = 10)
    
        # Else proceed to predict using CB model
        else:
            recommendation_items = model_basic(df_medpar, df_rental, df_sponsor)

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

# cat: category of the data (rentals/sponsorship/media_partner)
# cat_id: FK of the category of the data (rt_id/sp_id/mp_id)
# review_id: FK of the user who reviewed the product or service of the category
@app.get('/nlpe/collective/{cat}/{cat_id}')
def nlpe_coll_sp(cat_id:str,
                 cat:str,
                 response: Response):
    # load preprocessed data and sentences
    nlp_db = cne.Load_Data_NLP()
    DF_PROCESS = nlp_db.load_data(f'{cat}_review')
    nlp_db.close_database()
    sentences = DF_PROCESS.loc[DF_PROCESS.iloc[:,1]==cat_id,'review'].values

    # if the there is no desired id in the data
    if cat_id not in DF_PROCESS.iloc[:,1].values:
        return_dict = {'status':404,
                       'message': 'Emotion analysis for user review has failed.',
                       'data':{f'{cat}_id':cat_id,
                               'emotion_portion':None},
                       'success': False,
                       'error': f'The review id {cat_id} is not available.'}
        raise HTTPException(status_code=404, detail=return_dict)
    
    # if the desired id never have any review
    if len(sentences) == 0:
        return_dict = {'status':200,
                       'message':'No review for the id.',
                       'data':{f'{cat}_id':cat_id,
                               'emotion_portion':None},
                       'success': True,
                       'error': None}
        return JSONResponse(content=return_dict, status_code=200)
    
    else:
        # import the directory of the model, and encoder
        nlp_utils_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nlp')
        model_dir = os.path.join(nlp_utils_dir,'nlp_emotion_model.h5')
        encoder_dir = os.path.join(nlp_utils_dir,'label_encoder.joblib')
        tokenizer_dir = os.path.join(nlp_utils_dir,'my_tokenizer.json')

        # emotion analysis process
        nlp_class = cne.NLP_emotion(model_dir,encoder_dir,tokenizer_dir,sentences)
        nlp_por = nlp_class.percentage_emotions()

        return_dict = {'status':200,
                       'message':'Emotion analysis for user review has been successfully get.',
                       'data':{f'{cat}_id':cat_id,
                               'emotion_portion':nlp_por},
                       'success': True,
                       'error': None}
        return JSONResponse(content=return_dict, status_code=200)
    
@app.get('/nlpe/per_review/{cat}/{review_id}')
def nlpe_per_sp(review_id:str,
                cat:str,
                response:Response):
    # load preprocessed data and sentences
    nlp_db = cne.Load_Data_NLP()
    DF_PROCESS = nlp_db.load_data(f'{cat}_review')
    nlp_db.close_database()
    sentences = DF_PROCESS.loc[DF_PROCESS.iloc[:,0]==review_id,'review'].values

    # if the there is no desired id in the data
    if review_id not in DF_PROCESS.iloc[:,0].values:
        return_dict = {'status':404,
                       'message': 'Emotion analysis for user review has failed.',
                       'data':{'review_id':review_id,
                               'sentences': None,
                               'dominant_emotion':None,
                               'dominant_percentage':None},
                       'success': False,
                       'error': f'The review id {review_id} is not available.'}
        raise HTTPException(status_code=404, detail=return_dict)
    
    # if the desired id never have any review
    if len(sentences) == 0:
        return_dict = {'status':200,
                       'message':'No review for the id.',
                       'data':{'review_id':review_id,
                               'sentences': None,
                               'dominant_emotion':None,
                               'dominant_percentage':None},
                       'success': True,
                       'error': None}
        return JSONResponse(content=return_dict, status_code=200)
    else:
        # import the directory of the model, and encoder
        nlp_utils_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nlp')
        model_dir = os.path.join(nlp_utils_dir,'nlp_emotion_model.h5')
        encoder_dir = os.path.join(nlp_utils_dir,'label_encoder.joblib')
        tokenizer_dir = os.path.join(nlp_utils_dir,'my_tokenizer.json')

        # emotion analysis process
        nlp_class = cne.NLP_emotion(model_dir,encoder_dir,tokenizer_dir,sentences)
        nlp_por = nlp_class.percentage_emotions()
        max_key, max_value = max(nlp_por.items(), key=lambda x: x[1])

        return_dict = {'status':200,
                       'message':'Emotion analysis for user review has been successfully get.',
                       'data':{'review_id':review_id,
                               'sentences': str(sentences[0]),
                               'dominant_emotion':max_key,
                               'dominant_percentage':max_value},
                       'success': True,
                       'error': None}
        return JSONResponse(content=return_dict, status_code=200)

@app.get('/cb-recsys/{cat}/{cat_id}')
def cb_recsys(cat_id:str,
              cat:str=['sponsorship','rentals','media_partner']):
    # load preprocessed data
    cb_db = cb.Load_Data_CB()
    DF_PROCESS = cb_db.load_data(table=cat)
    cb_db.close_database()

    # if the there is no desired id in the data
    if cat_id not in DF_PROCESS['id'].values:
        return_dict = {'status':404,
                       'message': 'Recommendation for user has failed.',
                       'data': {f'{cat}_id':cat_id,
                                'recommendation':None},
                       'success': False,
                       'error': f'The id is not available in the {cat} table.'}
        raise HTTPException(status_code=404, detail=return_dict)
    
    else:
        # import the directory of the model, and encoder
        model_utils_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'recsys','model')
        encoder_path = os.path.join(model_utils_dir, f'{cat}_encoder.joblib')
        model_path = os.path.join(model_utils_dir, f'{cat}_cb_model.h5')

        # recommendation process
        recsys_model = cb.RecommenderSystem(DF_PROCESS, content_col="metadata",
                                            encoder_path=encoder_path,
                                            model_path=model_path)
        recsys_model.fit()
        id_recsys_array = recsys_model.recommend(cat_id).tolist()
        return_dict = {'status':200,
                       'message':'Recommendation for user has been successfully get.',
                       'data': {f'{cat}_id':cat_id,
                                'recommendation':id_recsys_array},
                       'success': True,
                       'error': None}
        return JSONResponse(content=return_dict, status_code=200) 