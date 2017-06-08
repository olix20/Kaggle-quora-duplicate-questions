from __future__ import print_function


import os
import re
import csv
import codecs
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from string import punctuation

from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.layers.wrappers import Bidirectional

import json
import bcolz
# import sys
# reload(sys)



########################################
## set directories and parameters
########################################
BASE_DIR = 'data/'
EMBEDDING_FILE = BASE_DIR + 'cache/GoogleNews-vectors-negative300.bin'
TRAIN_DATA_FILE = BASE_DIR + 'train.csv'
TEST_DATA_FILE = BASE_DIR + 'test.csv'
MAX_SEQUENCE_LENGTH = 35
MAX_NB_WORDS = 200000
EMBEDDING_DIM = 300





path = '/home/ubuntu/quora/'
data_home = path +"data/"



Q1_TRAINING_DATA_FILE = data_home+'cache/q1_train.npy'
Q2_TRAINING_DATA_FILE = data_home+'cache/q2_train.npy'
LABEL_TRAINING_DATA_FILE = data_home+'cache/label_train.npy'
WORD_EMBEDDING_MATRIX_FILE = data_home+'cache/word_embedding_matrix.npy'
NB_WORDS_DATA_FILE = data_home+'cache/nb_words.json'
Q1_TESTING_DATA_FILE = 'q1_test.npy'
Q2_TESTING_DATA_FILE = 'q2_test.npy'



data_1 = np.load(open(Q1_TRAINING_DATA_FILE, 'rb'))
data_2 = np.load(open(Q2_TRAINING_DATA_FILE, 'rb'))
labels = np.load(open(LABEL_TRAINING_DATA_FILE, 'rb'))
embedding_matrix = np.load(open(WORD_EMBEDDING_MATRIX_FILE, 'rb'))
with open(NB_WORDS_DATA_FILE, 'r') as f:
    nb_words = json.load(f)['nb_words']




test_data_1 = np.load(open(data_home+"cache/"+Q1_TESTING_DATA_FILE, 'rb'))
test_data_2 = np.load(open(data_home+"cache/"+Q2_TESTING_DATA_FILE, 'rb'))

df_test = pd.read_csv(data_home+'test.csv')



def save_array(fname, arr):
    c=bcolz.carray(arr, rootdir=fname, mode='w')
    c.flush()


def load_array(fname):
    return bcolz.open(fname)[:]




def train_and_predict():


    VALIDATION_SPLIT = 0.1

    num_lstm = 175#np.random.randint(175, 275)
    num_dense = 107#np.random.randint(100, 150)
    rate_drop_lstm = 0.2#0.15 + np.random.rand() * 0.25
    rate_drop_dense = 0.4# 0.15 + np.random.rand() * 0.25

    act = 'relu'
    re_weight = True # whether to re-weight classes to fit the 17.5% share in test set

    STAMP = 'main_model_lstm_%d_%d_%.2f_%.2f'%(num_lstm, num_dense, rate_drop_lstm, \
            rate_drop_dense)


    ########################################
    ## sample train/validation data
    ########################################
    #np.random.seed(1234)
    perm = np.random.permutation(len(data_1))
    idx_train = perm[:int(len(data_1)*(1-VALIDATION_SPLIT))]
    idx_val = perm[int(len(data_1)*(1-VALIDATION_SPLIT)):]

    data_1_train = np.vstack((data_1[idx_train], data_2[idx_train]))
    data_2_train = np.vstack((data_2[idx_train], data_1[idx_train]))
    labels_train = np.concatenate((labels[idx_train], labels[idx_train]))

    data_1_val = np.vstack((data_1[idx_val], data_2[idx_val]))
    data_2_val = np.vstack((data_2[idx_val], data_1[idx_val]))
    labels_val = np.concatenate((labels[idx_val], labels[idx_val]))

    weight_val = np.ones(len(labels_val))
    if re_weight:
        weight_val *= 0.472001959
        weight_val[labels_val==0] = 1.309028344




    embedding_layer = Embedding(nb_words+1,
            EMBEDDING_DIM,
            weights=[embedding_matrix],
            input_length=MAX_SEQUENCE_LENGTH,
            trainable=False)


    lstm_layer = LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm)

    sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_1 = embedding_layer(sequence_1_input)
    x1 = lstm_layer(embedded_sequences_1)

    sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_2 = embedding_layer(sequence_2_input)
    y1 = lstm_layer(embedded_sequences_2)


    concat_layer = concatenate([x1, y1])
    merged = Dropout(rate_drop_dense)(concat_layer)
    merged = BatchNormalization()(merged)

    # merged = Dense(num_dense, activation=act)(merged)
    # merged = Dropout(rate_drop_dense)(merged)
    # merged = BatchNormalization()(merged)

    merged = Dense(num_dense, activation=act)(merged)
    merged = Dropout(rate_drop_dense)(merged)
    merged = BatchNormalization()(merged)


    preds = Dense(1, activation='sigmoid')(merged)



    ########################################
    ## add class weight
    ########################################
    if re_weight:
        class_weight = {0: 1.309028344, 1: 0.472001959}
    else:
        class_weight = None

    ########################################
    ## train the model
    ########################################
    model = Model(inputs=[sequence_1_input, sequence_2_input], \
            outputs=preds)
    model.compile(loss='binary_crossentropy',
            optimizer='nadam',
            metrics=['acc'])
    #model.summary()

    print(STAMP)





    early_stopping =EarlyStopping(monitor='val_loss', patience=4)
    bst_model_path = "weights/"+STAMP+ '.h5'
    model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)

    # hist = model.fit([data_1_train, data_2_train], labels_train, \
    #         validation_data=([data_1_val, data_2_val], labels_val, weight_val), \
    #         epochs=200, batch_size=2048, shuffle=True, \
    #         class_weight=class_weight, callbacks=[early_stopping, model_checkpoint])



    # bst_val_score = min(hist.history['val_loss'])

    # print ("bst_val_score: ", bst_val_score )



    ########################################
    ## make the submission
    ########################################
    print('Start making the submission before fine-tuning')
    model.load_weights(bst_model_path)

    # preds = model.predict([test_data_1, test_data_2], batch_size=8192, verbose=1)
    # preds += model.predict([test_data_2, test_data_1], batch_size=8192, verbose=1)
    # preds /= 2

    # submission = pd.DataFrame({'test_id':df_test['test_id'], 'is_duplicate':preds.ravel()})
    # submission.to_csv('subm/forum_bilstm_%.4f_'%(bst_val_score)+STAMP+'.csv', index=False)


    # this is the way to reuse features 
    features_model =Model(inputs=[sequence_1_input, sequence_2_input], outputs=concat_layer)
    features_model.compile(loss='mse', optimizer='adam')

    print ("predicting lstm features for training data ")

    #creating training features 
    lstm_features_train_q1_q2 = features_model.predict([data_1, data_2], batch_size=8192)
    lstm_features_train_q2_q1 = features_model.predict([data_2, data_1], batch_size=8192)

    save_array("data/cache/lstm_features_train_q1_q2.dat",lstm_features_train_q1_q2)
    save_array("data/cache/lstm_features_train_q2_q1.dat",lstm_features_train_q2_q1)


    print ("predicting lstm features for test data ")

    #creating test features 

    lstm_features_test_q1_q2 = features_model.predict([test_data_1, test_data_2], batch_size=8192)
    lstm_features_test_q2_q1 = features_model.predict([test_data_2, test_data_1], batch_size=8192)
    save_array("data/cache/lstm_features_test_q1_q2.dat",lstm_features_test_q1_q2)
    save_array("data/cache/lstm_features_test_q2_q1.dat",lstm_features_test_q2_q1)



train_and_predict()