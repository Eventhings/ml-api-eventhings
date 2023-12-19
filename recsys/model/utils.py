import os
import math
import heapq
import pandas as pd
import numpy as np
import psycopg2

from sklearn.preprocessing import LabelEncoder

#------------------------------------------------------------------------------------------------------------------------------#

from dotenv import load_dotenv
import os

def get_negatives(uids, iids, items, df_test):
    """
    Returns a pandas dataframe of 20 negative interactions
    based for each user in df_test.
    
    Args:
        uids (np.array): Numpy array of all user ids.
        iids (np.array): Numpy array of all item ids.
        items (list): List of all unique items.
        df_test (dataframe): Our test set.
        
    Returns:
        df_neg (dataframe): dataframe with 100 negative items 
            for each (u, i) pair in df_test.
    """

    negativeList = []
    test_u = df_test['user_id'].values.tolist()
    test_i = df_test['vendor_id'].values.tolist()

    test_ratings = list(zip(test_u, test_i))
    zipped = set(zip(uids, iids))

    for (u, i) in test_ratings:
        negatives = []
        negatives.append((u, i))
        for t in range(20):
            j = np.random.randint(len(items)) # Get random item id.
            while (u, j) in zipped: # Check if there is an interaction
                j = np.random.randint(len(items)) # If yes, generate a new item id
            negatives.append(j) # Once a negative interaction is found we add it.
        negativeList.append(negatives)

    df_neg = pd.DataFrame(negativeList)

    return df_neg


def mask_first(x):
    """
    Return a list of 0 for the first item and 1 for all others
    """

    result = np.ones_like(x)
    result[0] = 0
    
    return result
   

def train_test_split(df):
    """
    Splits our original data into one test and one
    training set. 
    The test set is made up of one item for each user. This is
    our holdout item used to compute Top@K later.
    The training set is the same as our original data but
    without any of the holdout items.
    
    Args:
        df (dataframe): Our original data
        
    Returns:
        df_train (dataframe): All of our data except holdout items
        df_test (dataframe): Only our holdout items.
    """

    # Create two copies of our dataframe that we can modify
    df_test = df.copy(deep = True)
    df_train = df.copy(deep = True)

    # Group by user_id and select only the first item for
    # each user (our holdout).
    df_test = df_test.groupby(['user_id']).first()
    df_test['user_id'] = df_test.index
    df_test = df_test[['user_id', 'vendor_id', 'rating']]

    # Remove the same items for our test set in our training set.
    mask = df.groupby(['user_id'])['user_id'].transform(mask_first).astype(bool)
    df_train = df.loc[mask]

    return df_train, df_test


def load_dataset():
    """
    Loads the dataset and transforms it into the format we need. 
    We then split it into a training and a test set.
    """

    try:
        dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
        load_dotenv(dotenv_path)
        DB_USERNAME = os.getenv('DB_USERNAME')
        DB_PASSWORD = os.getenv('DB_PASSWORD')
        DB_HOST = os.getenv('DB_HOST')
        DB_PORT = os.getenv('DB_PORT')
        DB_DATABASE = os.getenv('DB_DATABASE')
        DB_DATABASE_DEV = os.getenv('DB_DATABASE_DEV')

        connection = psycopg2.connect(database = DB_DATABASE, 
                                      host = DB_HOST, 
                                      user = DB_USERNAME,
                                      password = DB_PASSWORD, 
                                      port = DB_PORT)
        
        cursor = connection.cursor()

        # Fetch data for media partner
        cursor.execute("SELECT user_id, mp_id, rating FROM media_partner_review")
        medpar_data = cursor.fetchall()
        medpar_cols = [desc.name for desc in cursor.description]
        df_medpar = pd.DataFrame(medpar_data, columns = medpar_cols)
        df_medpar = df_medpar.rename(columns = {'mp_id': 'vendor_id'})

        # Fetch data for rental
        cursor.execute("SELECT user_id, rt_id, rating FROM rentals_review")
        rental_data = cursor.fetchall()
        rental_cols = [desc.name for desc in cursor.description]
        df_rental = pd.DataFrame(rental_data, columns = rental_cols)
        df_rental = df_rental.rename(columns = {'rt_id': 'vendor_id'})

        # Fetch data for sponsorship
        cursor.execute("SELECT user_id, sp_id, rating FROM sponsorship_review")
        sponsor_data = cursor.fetchall()
        sponsor_cols = [desc.name for desc in cursor.description]
        df_sponsor = pd.DataFrame(sponsor_data, columns = sponsor_cols)
        df_sponsor = df_sponsor.rename(columns = {'sp_id': 'vendor_id'})

        # Union the results
        df = pd.concat([df_medpar, df_rental, df_sponsor], ignore_index=True)
        df = df[['user_id', 'vendor_id', 'rating']]

        # Encode user_id and vendor_id
        label_encoder_user = LabelEncoder()
        label_encoder_item = LabelEncoder() 

        df['user_id'] = label_encoder_user.fit_transform(df['user_id'])
        df['vendor_id'] = label_encoder_item.fit_transform(df['vendor_id'])

        # Create training and test sets.
        df_train, df_test = train_test_split(df)

        # Create lists of all unique users and artists
        users = list(df['user_id'].unique()) # Label
        items = list(df['vendor_id'].unique()) # Label

        # Get all user ids and item ids.
        uids = df_train['user_id'].tolist() # Label
        iids = df_train['vendor_id'].tolist() # Label

        # Sample negative interactions for each user in our test data
        df_neg = get_negatives(uids, iids, items, df_test)

        return uids, iids, df_train, df_test, df_neg, users, items, label_encoder_user, label_encoder_item
    
    # Exception handling
    except Exception as e:
        return e
    
    # Close DB connection
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


def get_train_instances(uids, iids, items, num_neg = 5):
    """
    Samples a number of negative user-item interactions for each
    user-item pair in our testing data.

    Args:
        uids (list): All users' ID within the historical data
        iids (list): All items' ID within the historical data
        items (list): List of all unique items
     
    Returns:
        user_input (list): A list of all users for each item
        item_input (list): A list of all items for every user,
            both positive and negative interactions.
        labels (list): A list of all labels. 0 or 1.
    """

    user_input, item_input, labels = [], [], []
    zipped = set(zip(uids, iids))

    for (u, i) in zip(uids, iids):
        # Add our positive interaction
        user_input.append(u)
        item_input.append(i)
        labels.append(1)

        # Sample a number of random negative interactions
        for t in range(num_neg):
            j = np.random.randint(len(items))
            while (u, j) in zipped:
                j = np.random.randint(len(items))
            user_input.append(u)
            item_input.append(j)
            labels.append(0)

    return user_input, item_input, labels


def get_hits(k_ranked, holdout):
    """
    Return 1 if the holdout item exists in a given list and 0 if not.
    """

    for item in k_ranked:
        if item == holdout:
            return 1
    return 0


def get_ndcg(ranklist, gtItem):
    """
    NDCG@k is a ranking metric that helps consider 
    both the relevance of items and their positions in the list.
    """

    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i+2)
    return 0


def predict_ratings_cf(user_id, items, model, label_user, label_item):
    """
    Predict rating score for each user.

    Args:
        user_id (int): Current user_id
        items (list): List of all unique items
        model (h5): Tensorflow model
        label_user (int): Label encoder for user
        label_item (int): Label encoder for item
        
    Returns:
        map_item_score (list): predicted current user rating for each item.
    """

    # Prepare user and item arrays for the model.
    user_id = label_user.transform([user_id])[0]
    predict_user = np.full(len(items), user_id, dtype = 'int32').reshape(-1, 1)
    np_items = np.array(items).reshape(-1, 1)

    # Predict ratings using the model.
    predictions = model.predict([predict_user, np_items]).flatten().tolist()

    # Map predicted score to item id.
    items = label_item.inverse_transform(items)
    map_item_score = dict(zip(items, predictions))

    return map_item_score
    

def eval_rating(idx, test_ratings, test_negatives, K, model, label_user, label_item):
    """
    Generate ratings for the users in our test set and
    check if our holdout item is among the top K highest scores
    and evaluate its position.
    
    Args:
        idx (int): Current user_id label
        test_ratings (list): Our test set user-item pairs
        test_negatives (list): negative items for each
            user in our test set.
        K (int): number of top recommendations
        label_user (int): Label encoder for user
        label_item (int): Label encoder for item
        
    Returns:
        hitrate (list): A list of 1 if the holdout appeared in our
            top K predicted items. 0 if not.
    """

    # Get the negative interactions for our user.
    items = test_negatives[idx]

    # Get the user idx.
    user_idx = test_ratings[idx][0]
    user_idx = label_user.inverse_transform([user_idx])[0]

    # Get the item idx, i.e., our holdout item.
    holdout = test_ratings[idx][1]

    # Add the holdout to the end of the negative interactions list.
    items.append(holdout)

    # Predict ratings using the model.
    map_item_score = predict_ratings_cf(user_idx, items, model, label_user, label_item)
    items.pop()

    # Get the K highest ranked items as a list.
    k_ranked = heapq.nlargest(K, map_item_score, key = map_item_score.get)

    # Get a list of hit or no hit.
    hitrate = get_hits(k_ranked, holdout)
    ndcg = get_ndcg(k_ranked, holdout)

    return (hitrate, ndcg)


def evaluate(model, df_test, df_neg, label_user, label_item, K = 10):
    """
    Calculate the top@K hit ratio for our recommendations.
    
    Args:
        df_neg (dataframe): dataframe containing our holdout items
            and 100 randomly sampled negative interactions for each
            (user, item) holdout pair.
        label_user (int): Label encoder for user
        label_item (int): Label encoder for item
        K (int): The 'K' number of ranked predictions we want
            our holdout item to be present in. 
            
    Returns:
        hits (list): list of "hits". 1 if the holdout was present in 
            the K highest ranked predictions. 0 if not. 
    """

    hitrates = []
    ndcgs = []

    test_u = df_test['user_id'].values.tolist()
    test_i = df_test['vendor_id'].values.tolist()

    test_ratings = list(zip(test_u, test_i))

    df_neg = df_neg.drop(df_neg.columns[0], axis = 1)
    test_negatives = df_neg.values.tolist()

    for user_id in test_u:
        # For each idx, call eval_one_rating
        (hitrate, ndcg) = eval_rating(user_id, test_ratings, test_negatives, K, model, label_user, label_item)
        hitrates.append(hitrate)
        ndcgs.append(ndcg)

    return (hitrates, ndcgs)


def get_top_k_items(dct, k = 10):
    # Use nlargest to get the top n key-value pairs based on values.
    top_k_items = heapq.nlargest(k, dct.items(), key = lambda item: item[1])

    # Calculate avg rating for each item.
    average_ratings = [item[1] for item in top_k_items]
    recommendation_items = [item[0] for item in top_k_items]

    # Convert all items to strings.
    average_ratings = [str(item) for item in average_ratings]
    recommendation_items = [str(item) for item in recommendation_items]

    return average_ratings, recommendation_items