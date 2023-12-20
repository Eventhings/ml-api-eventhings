import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Flatten, Multiply, Dropout, Dense, BatchNormalization, Concatenate
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L1, L2

#------------------------------------------------------------------------------------------------------------------------------#

def model_basic(df_medpar, df_rental, df_sponsor):
    """
    Load data from cloud database
    """

    # Sort values for all data based on their average rating and get their top 10 items
    df_medpar = df_medpar.groupby('vendor_id')['rating'].mean().reset_index()
    df_rental = df_rental.groupby('vendor_id')['rating'].mean().reset_index()
    df_sponsor = df_sponsor.groupby('vendor_id')['rating'].mean().reset_index()

    # Get a random number of top items for each category, ensuring the total is 10
    total_top_items = 10
    top_items_medpar = np.random.randint(3, 5)  # Randomly choose 3 or 4 top items
    top_items_rental = np.random.randint(3, 5)  # Randomly choose 3 or 4 top items
    top_items_sponsor = total_top_items - top_items_medpar - top_items_rental

    # Select the top items for each category
    df_medpar = df_medpar.sort_values(by = 'rating', ascending = False).iloc[:top_items_medpar]
    df_rental = df_rental.sort_values(by = 'rating', ascending = False).iloc[:top_items_rental]
    df_sponsor = df_sponsor.sort_values(by = 'rating', ascending = False).iloc[:top_items_sponsor]

    # Concat the results then shuffle
    df = pd.concat([df_medpar, df_rental, df_sponsor])
    df = df[['vendor_id']].values
    recommendations = []
    for item in df:
        recommendations.append(item[0])
    np.random.shuffle(recommendations)

    return recommendations


def model_cf(users, items):
    """
    Tensorflow collaborative-filtering recsys model for the mainpage within the app.

    Args:
        users (list): all unique users within historical data.
        items (list): all unique items within historical data.
    """
    
    # HYPERPARAMS
    latent_features = 20
    learning_rate = 0.005

    # TENSORFLOW GRAPH
    # Using the functional API

    # Define input layers for user, item, and label.
    user_input = Input(shape = (1,), dtype = tf.int32, name = 'user')
    item_input = Input(shape = (1,), dtype = tf.int32, name = 'item')
    label_input = Input(shape = (1,), dtype = tf.int32, name = 'label')

    # User embedding for MLP
    mlp_user_embedding = Embedding(input_dim = len(users), 
                                    output_dim = latent_features,
                                    embeddings_initializer = 'random_normal',
                                    embeddings_regularizer = L1(0.01),
                                    input_length = 1, 
                                    name = 'mlp_user_embedding')(user_input)

    # Item embedding for MLP
    mlp_item_embedding = Embedding(input_dim = len(items), 
                                   output_dim = latent_features,
                                   embeddings_initializer = 'random_normal',
                                   embeddings_regularizer = L1(0.01),
                                   input_length = 1, 
                                   name = 'mlp_item_embedding')(item_input)

    # User embedding for GMF
    gmf_user_embedding = Embedding(input_dim = len(users), 
                                   output_dim = latent_features,
                                   embeddings_initializer = 'random_normal',
                                   embeddings_regularizer = L1(0.01),
                                   input_length = 1, 
                                   name = 'gmf_user_embedding')(user_input)

    # Item embedding for GMF
    gmf_item_embedding = Embedding(input_dim = len(items), 
                                   output_dim = latent_features,
                                   embeddings_initializer = 'random_normal',
                                   embeddings_regularizer = L1(0.01),
                                   input_length = 1, 
                                   name = 'gmf_item_embedding')(item_input)

    # GMF layers
    gmf_user_flat = Flatten()(gmf_user_embedding)
    gmf_item_flat = Flatten()(gmf_item_embedding)
    gmf_matrix = Multiply()([gmf_user_flat, gmf_item_flat])

    # MLP layers
    mlp_user_flat = Flatten()(mlp_user_embedding)
    mlp_item_flat = Flatten()(mlp_item_embedding)
    mlp_concat = Concatenate()([mlp_user_flat, mlp_item_flat])

    mlp_dropout = Dropout(0.1)(mlp_concat)

    mlp_layer_1 = Dense(64, 
                        activation = 'relu', 
                        name = 'mlp_layer1')(mlp_dropout)
    mlp_batch_norm1 = BatchNormalization(name = 'mlp_batch_norm1')(mlp_layer_1)
    mlp_dropout1 = Dropout(0.1, 
                        name = 'mlp_dropout1')(mlp_batch_norm1)

    mlp_layer_2 = Dense(32, 
                        activation = 'relu', 
                        name = 'mlp_layer2')(mlp_dropout1)
    mlp_batch_norm2 = BatchNormalization(name = 'mlp_batch_norm2')(mlp_layer_2)
    mlp_dropout2 = Dropout(0.1, 
                        name = 'mlp_dropout2')(mlp_batch_norm2)

    mlp_layer_3 = Dense(16, 
                        activation = 'relu', 
                        kernel_regularizer = L2(0.01),
                        name = 'mlp_layer3')(mlp_dropout2)
    mlp_layer_4 = Dense(8, 
                        activation = 'relu', 
                        activity_regularizer = L2(0.01),
                        name = 'mlp_layer4')(mlp_layer_3)

    # Merge the two networks
    merged_vector = Concatenate()([gmf_matrix, mlp_layer_4])

    # Output layer
    output_layer = Dense(1, 
                        activation = 'sigmoid',
                        kernel_initializer = 'lecun_uniform',
                        name = 'output_layer')(merged_vector)

    # Define the model
    model = Model(inputs = [user_input, item_input], outputs = output_layer)

    # Compile the model with binary cross entropy loss and Adam optimizer
    optimizer = Adam(learning_rate = learning_rate)
    model.compile(optimizer = optimizer,
                  loss = 'binary_crossentropy',
                  metrics = ['accuracy'])
    
    return model