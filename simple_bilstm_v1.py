from __future__ import print_function
import numpy as np
import pandas as pd
import datetime, time, json
import os.path
import pickle 

from keras import backend as K

from keras.models import Sequential,Model
from keras.layers import Embedding, Dense, Dropout, Reshape, Merge, BatchNormalization, Lambda ,merge
from keras.layers import Conv1D , Flatten, Input
from keras.layers.pooling import MaxPooling1D
from keras.layers.convolutional import ZeroPadding1D
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.layers.recurrent import LSTM

from keras.layers.merge import * 
from keras.layers.pooling import GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.optimizers import Adam

from keras.regularizers import l2
from keras.callbacks import Callback, ModelCheckpoint
from keras.initializers import he_normal



# from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit






path = '/home/ubuntu/quora/'
data_home = path +"data/"

Q1_TRAINING_DATA_FILE = data_home+'cache/q1_train.npy'
Q2_TRAINING_DATA_FILE = data_home+'cache/q2_train.npy'
LABEL_TRAINING_DATA_FILE = data_home+'cache/label_train.npy'
WORD_EMBEDDING_MATRIX_FILE = data_home+'cache/word_embedding_matrix.npy'
NB_WORDS_DATA_FILE = data_home+'cache/nb_words.json'
Q1_TESTING_DATA_FILE = 'q1_test.npy'
Q2_TESTING_DATA_FILE = 'q2_test.npy'


MODEL_WEIGHTS_FILE = path+'weights/conv_weights_v1.h5'
MAX_SEQUENCE_LENGTH = 35
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.1
TEST_SPLIT = 0.1
RNG_SEED = 13371447
NB_EPOCHS = 25

nr_hidden = 100
drop_out=0.2


word_embedding_matrix = np.load(open(WORD_EMBEDDING_MATRIX_FILE, 'rb'))
with open(NB_WORDS_DATA_FILE, 'r') as f:
    nb_words = json.load(f)['nb_words']


def prepare_data():
	q1_data = np.load(open(Q1_TRAINING_DATA_FILE, 'rb'))
	q2_data = np.load(open(Q2_TRAINING_DATA_FILE, 'rb'))
	labels = np.load(open(LABEL_TRAINING_DATA_FILE, 'rb'))



	sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=2019)
	X = np.stack((q1_data, q2_data), axis=1)
	y = labels

	for train_index, test_index in sss.split(X, y):
	    print("TRAIN:", train_index, "TEST:", test_index)
	    X_train, X_test = X[train_index], X[test_index]
	    y_train, y_test = y[train_index], y[test_index]



	# Q1_train = X_train[:,0]
	# Q2_train = X_train[:,1]
	# Q1_test = X_test[:,0]
	# Q2_test = X_test[:,1]

	# return Q1_train,Q2_train,Q1_test,Q2_test
	return X_train,X_test,y_train,y_test

def create_bilstm():

    
    graph_in = Input(shape=(MAX_SEQUENCE_LENGTH,EMBEDDING_DIM))
    
    
    Q = Sequential()
    
    # Embed
    Q.add(Embedding(nb_words + 1, 
                     EMBEDDING_DIM, 
                     weights=[word_embedding_matrix], 
                     input_length=MAX_SEQUENCE_LENGTH, 
                     trainable=False,
                    input_shape=(MAX_SEQUENCE_LENGTH,)))


    # Encode 
    Q.add(Bidirectional(LSTM(nr_hidden, return_sequences=False,
                                             dropout=drop_out, recurrent_dropout=drop_out)))    
    
    return Q



def createModel():

	q1 = Input(shape=(MAX_SEQUENCE_LENGTH,))
	q2 = Input(shape=(MAX_SEQUENCE_LENGTH,))#,EMBEDDING_DIM

	encoder = create_bilstm()

	q1_encoded = encoder(q1)
	q2_encoded = encoder(q2)


	regressor_model = Sequential()

	# regressor_model.add(Dropout(drop_out, input_shape=(nr_hidden*2,)))
	regressor_model.add(Dense(200, name='entail1',kernel_initializer='he_normal',activation='relu', input_shape=(nr_hidden*4,)))
	regressor_model.add(Dropout(drop_out))
	regressor_model.add(BatchNormalization())

	regressor_model.add(Dense(200, name='entail2', kernel_initializer='he_normal',activation='relu'))
	regressor_model.add(Dropout(drop_out))
	regressor_model.add(BatchNormalization())

	regressor_model.add(Dense(200, name='entail3', kernel_initializer='he_normal',activation='relu'))
	regressor_model.add(Dropout(drop_out))
	regressor_model.add(BatchNormalization())


	regressor_model.add(Dense(200, name='entail4', kernel_initializer='he_normal',activation='relu'))
	regressor_model.add(Dropout(drop_out))
	regressor_model.add(BatchNormalization())

	regressor_model.add(Dense(1, name='entail_out', activation='sigmoid'))

	scores = regressor_model(concatenate([q1_encoded, q2_encoded]))


	model = Model(inputs=[q1, q2], outputs=[scores])



	return model




def trainModel(model,X_train,X_test,y_train,y_test):

	Q1_train = X_train[:,0]
	Q2_train = X_train[:,1]
	Q1_test = X_test[:,0]
	Q2_test = X_test[:,1]





	model.compile(loss='binary_crossentropy', 
              optimizer="nadam", 
              metrics=['accuracy'])#, 'precision', 'recall', 'fbeta_score'])


	MODEL_WEIGHTS_FILE = path+'weights/simple_bilstm_v1_epoch_{epoch:02d}_val_loss_{val_loss:.2f}.h5'

	#in case of broken training epochs
	if os.path.isfile(path+"simple_bilstm_v1_00-0.46.h5"): 
		model.load_weights(path+"simple_bilstm_v1_00-0.46.h5")

	callbacks = [ModelCheckpoint(MODEL_WEIGHTS_FILE, monitor='val_loss', save_best_only=True)]


	print("Starting training at", datetime.datetime.now())
	t0 = time.time()




	history = model.fit([Q1_train, Q2_train],
	                    y_train,
	                    epochs=15,
	                    batch_size=128,
	                    # validation_split=VALIDATION_SPLIT,
	                    validation_data = ([Q1_test, Q2_test],y_test),
	                    callbacks=callbacks)


	t1 = time.time()
	print("Training ended at", datetime.datetime.now())
	print("Minutes elapsed: %f" % ((t1 - t0) / 60.))


	with open(path+'logs/simple_lstm_v1_1.txt','wb') as handle:
		pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)





if __name__ == "__main__":

	X_train,X_test,y_train,y_test = prepare_data()

	model = createModel()

	trainModel(model,X_train,X_test,y_train,y_test)

