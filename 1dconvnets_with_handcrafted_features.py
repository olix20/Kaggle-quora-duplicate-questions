from __future__ import print_function


import os
import re
import csv
import codecs
import numpy as np
import pandas as pd

# from nltk.corpus import stopwords
# from nltk.stem import SnowballStemmer
# from string import punctuation

# from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.layers.wrappers import Bidirectional
from sklearn.preprocessing import normalize

import json
from sklearn.preprocessing import StandardScaler
from utils import *


from tqdm import tqdm


# import sys
# reload(sys)

from keras.models import Sequential,Model
from keras.layers import Embedding, Dense, Dropout, Reshape, Merge, BatchNormalization, TimeDistributed, Lambda
from keras.layers import Conv1D , Flatten, Input
from keras.layers.pooling import MaxPooling1D
from keras.layers.pooling import GlobalMaxPooling1D



from keras.regularizers import l2
from keras.callbacks import Callback, ModelCheckpoint,EarlyStopping
from keras import backend as K






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

# df_test = pd.read_csv(data_home+'test.csv')


# extra features

train_features = load_array(data_home+"cache/train_features_extrasvd_v2.dat")
test_features = load_array(data_home+"cache/test_features_extrasvd_v2.dat")



train_features = np.nan_to_num(train_features)
test_features = np.nan_to_num(test_features)





num_folds = 5
X = train_features
X_submission = np.nan_to_num(StandardScaler().fit_transform(test_features)) 

y = labels





def get_convs():
#v2
    graph_in = Input(shape=(MAX_SEQUENCE_LENGTH,))
    embedded = Embedding(nb_words + 1, 
                 EMBEDDING_DIM, 
                 weights=[embedding_matrix], 
                 input_length=MAX_SEQUENCE_LENGTH, 
                 trainable=False)(graph_in)
    
#     normalized_input = BatchNormalization()(embedded)
    
    convs = []

    for fsz in range(1,4):
        conv = Conv1D(64, fsz,
                      padding = 'valid', activation='relu')(embedded)#
        conv = Dropout(0.3)(conv)
        conv = BatchNormalization()(conv)        
        
        pool = MaxPooling1D(pool_size=2)(conv)
        flatten = Flatten()(pool)
#         pool = GlobalMaxPooling1D()(conv)
        convs.append(flatten)

    out = concatenate(convs) 
    graph = Model(inputs=graph_in,outputs=out)

    return graph






def create_model(num_hidden1,num_hidden2,rate_drop_1,rate_drop_2):

    act = 'relu'

    STAMP = 'NN_%d_%d_%.2f_%.2f'%(num_hidden1, num_hidden2, rate_drop_1, \
            rate_drop_2)



    #model.summary()

    print(STAMP)




    graph = get_convs()

    q1_input = Input(shape=(MAX_SEQUENCE_LENGTH,))
    q2_input = Input(shape=(MAX_SEQUENCE_LENGTH,))

    Q1 = graph(q1_input)
    Q2 = graph(q2_input)


    q1q2 = concatenate([Q1, Q2])


    abhishek_features = Input((train_features.shape[1],),name="abhishek_features")




    model = concatenate([Q1,Q2,abhishek_features])
    model =Dropout(0.2)(model)

    model = BatchNormalization()(model)
    model =Dense(num_hidden1, activation='relu')(model)

    model =Dropout(rate_drop_1)(model)
    model =BatchNormalization()(model)
    model =Dense(num_hidden2, activation='relu')(model)


    model =Dropout(rate_drop_2)(model)
    model =BatchNormalization()(model)
    model =Dense(1, activation='sigmoid')(model)

    model = Model(inputs=[q1_input,q2_input,abhishek_features],outputs=model)

    model.compile(loss='binary_crossentropy',
        optimizer='nadam',
        metrics=['acc'])



    return model, STAMP




y_oof = np.zeros((train_features.shape[0],1))
test_preds = np.zeros((test_features.shape[0],1))
val_scores = []





early_stopping =EarlyStopping(monitor='val_loss', patience=4)





########################################
## train the model
########################################



num_iters = 1
for i in range(num_iters):


    ## create random params for current iteration
    num_hidden1 = np.random.randint(300, 400)
    num_hidden2 = np.random.randint(300, 400)
    rate_drop_1 = 0.2 + np.random.rand() * 0.15
    rate_drop_2 = 0.2 + np.random.rand() * 0.15




    num_folds = 5
    current_fold = 0 
    skf = StratifiedKFold(n_splits=num_folds, random_state=2019)
    splits = skf.split(train_features, labels)

    for ix_fit, ix_valid in tqdm(splits, total=num_folds):
        
        y_train = labels[ix_fit]
        y_val = labels[ix_valid]

        X_train = np.nan_to_num(StandardScaler().fit_transform(train_features[ix_fit]))
        X_valid = np.nan_to_num(StandardScaler().fit_transform(train_features[ix_valid]))



        data_1_train = data_1[ix_fit]
        data_2_train = data_2[ix_fit]

 
        data_1_val = data_1[ix_valid]
        data_2_val = data_2[ix_valid]




        current_fold +=1

    
        ##### create model
        model,STAMP =create_model(num_hidden1,num_hidden2,rate_drop_1,rate_drop_2)


        kfold_weights_path = "weights/NN_oof_{}_fold_{}.h5".format(STAMP,current_fold)
        callbacks = [EarlyStopping(monitor='val_loss', patience=4),
                     ModelCheckpoint(kfold_weights_path, monitor='val_loss', save_best_only=True)]




        # model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])



        history = model.fit(x = [data_1_train, data_2_train,X_train], y= y_train, 
            validation_data=([data_1_val, data_2_val,X_valid], y_val),
          batch_size=128, epochs=100,callbacks=callbacks,  shuffle=True,verbose=1)


        ########### evaluation

        bst_val_score = min(history.history['val_loss'])
        tqdm.write("bst_val_score:{}".format(bst_val_score) )
        val_scores.append(bst_val_score)



        tqdm.write('Start making the submission before fine-tuning')
        model.load_weights(kfold_weights_path)

        y_oof[ix_valid] += model.predict([data_1_val, data_2_val,X_valid], batch_size=8192, verbose=1).reshape(-1,1)

        test_preds += model.predict([test_data_1, test_data_2,X_submission], batch_size=8192, verbose=1).reshape(-1,1)
        # preds += model.predict([test_data_2, test_data_1,abhishek_np_test], batch_size=8192, verbose=1)
       




save_array("data/results/onedconvs_5fold_s2019_oof_preds_extrasvdv2.dat",(y_oof/float(num_iters)))
save_array("data/results/onedconvs_5fold_s2019_test_preds_extrasvdv2.dat",(test_preds/float(num_folds*num_iters)))
save_array("data/results/onedconvs_val_scores.dat",val_scores)


