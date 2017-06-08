from __future__ import print_function
import numpy as np
import pandas as pd
import datetime, time, json
import os.path
import pickle 

import keras 
from keras import backend as K

from keras.models import Sequential,Model
from keras.layers import Embedding, Dense, Dropout, Reshape, Merge, BatchNormalization, Lambda ,merge, Activation
from keras.layers import Conv1D , Flatten, Input
from keras.layers.pooling import MaxPooling1D
from keras.layers.convolutional import ZeroPadding1D
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.layers.recurrent import LSTM

from keras.layers.merge import * 
from keras.layers.pooling import GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.optimizers import Adam

from keras.regularizers import l2
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
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

JUST_TESTING = True


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
		# print("TRAIN:", train_index, "TEST:", test_index)
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]


	if JUST_TESTING: #working with 
		X = X_train
		y = y_train

		sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=2021)
		for train_index, test_index in sss.split(X , y):
			X_train = X[train_index]
			y_train = y[train_index]


	# Q1_train = X_train[:,0]
	# Q2_train = X_train[:,1]
	# Q1_test = X_test[:,0]
	# Q2_test = X_test[:,1]

	# return Q1_train,Q2_train,Q1_test,Q2_test
	return X_train,X_test,y_train,y_test


def create_graph():
	features = []

	
	graph_in = Input(shape=(MAX_SEQUENCE_LENGTH,)) #seems you can't miss this F*%!@$ layer
	
	
	embeddings = Embedding(nb_words + 1, 
				 EMBEDDING_DIM, 
				 weights=[word_embedding_matrix], 
				 input_length=MAX_SEQUENCE_LENGTH, 
				 trainable=False)(graph_in)

	init_batch = BatchNormalization()(embeddings)
	
	
	# series of 1d convnets 
	for fsz in range(3,6):
 
		conv = Conv1D(128, fsz,strides=1, # is it covering words or characters? words cause each row represents a word
							padding = 'valid', activation='relu')(init_batch)
	
	
		batch = BatchNormalization()(conv)
		pool = MaxPooling1D(pool_size=2)(batch)
		drop = Dropout(0.15)(pool)
		flatten = Flatten()(drop)
		features.append(flatten)
	
	## use lstm output as a feature
	lstm = LSTM(128, return_sequences=False,dropout=0.2, recurrent_dropout=0.2)(init_batch)
	features.append(lstm)
	
	
	# ## use residuals direct from word-to-vec embeddings (ResNet style)
	# dense1 = Dense(128)(init_batch)
	# bn1 = BatchNormalization()(dense1)
	# relu1 = Activation('relu')(bn1)
	# relu1 = Dropout(0.2)(relu1)
	
	# dense2 = Dense(128)(relu1)
	# bn2 = BatchNormalization()(dense2)
	# res2 = merge([relu1, bn2], mode='sum')
	# relu2 = Activation('relu')(res2)    
	# relu2 = Dropout(0.2)(relu2)
	
	
	# dense3 = Dense(128)(relu2)
	# bn3 = BatchNormalization()(dense3)
	# res3 = Merge(mode='sum')([relu2, bn3])
	# relu3 = Activation('relu')(res3)   
	# relu3 = Dropout(0.2)(relu3)
	
	
	
	# res_parts = keras.layers.concatenate([relu3, relu2, relu1])
	# bn4 = BatchNormalization()(res_parts)
	# flat_res = Flatten()(bn4)
	# features.append(flat_res)
	

	# put everything together
	out = keras.layers.concatenate(features) 
	graph = Model(inputs=graph_in,outputs=out)

	
	return graph

def createModel():


	q1_input = Input(shape=(MAX_SEQUENCE_LENGTH,))
	q2_input = Input(shape=(MAX_SEQUENCE_LENGTH,))


		

	_1d_convs_and_lstm = create_graph()


	### add computation graph to question 1 inputs

	Q1 = _1d_convs_and_lstm(q1_input)
	Q2 = _1d_convs_and_lstm(q2_input)



	model = keras.layers.concatenate([Q1, Q2])
	model = Dropout(0.15)(model)
	model = BatchNormalization()(model)

	model = Dense(200, activation='relu')(model)
	model = Dropout(0.15)(model)
	model = BatchNormalization()(model)

	model = Dense(200, activation='relu')(model)
	model = Dropout(0.15)(model)
	model = BatchNormalization()(model)

	model = Dense(1, activation='sigmoid')(model)

	model = Model(inputs=[q1_input,q2_input],outputs=model)

	return model




def trainModel(model,X_train,X_test,y_train,y_test):

	Q1_train = X_train[:,0]
	Q2_train = X_train[:,1]
	Q1_test = X_test[:,0]
	Q2_test = X_test[:,1]





	model.compile(loss='binary_crossentropy', 
			  optimizer="nadam", 
			  metrics=['accuracy'])#, 'precision', 'recall', 'fbeta_score'])


	MODEL_WEIGHTS_FILE = path+'weights/bilstm_and_1dconvnets_v1_epoch_{epoch:02d}_val_loss_{val_loss:.2f}.h5'

	#in case of broken training epochs
	# if os.path.isfile(path+"simple_bilstm_v1_00-0.46.h5"): 
	# 	model.load_weights(path+"simple_bilstm_v1_00-0.46.h5")

	early_stopping =EarlyStopping(monitor='val_loss', patience=4)
	callbacks = [ModelCheckpoint(MODEL_WEIGHTS_FILE, monitor='val_loss', save_best_only=True),early_stopping]


	print("Starting training at", datetime.datetime.now())
	t0 = time.time()




	history = model.fit([Q1_train, Q2_train],
						y_train,
						epochs=45,
						batch_size=512,
						shuffle=True,
						# validation_split=VALIDATION_SPLIT,
						validation_data = ([Q1_test, Q2_test],y_test),
						callbacks=callbacks)


	t1 = time.time()
	print("Training ended at", datetime.datetime.now())
	print("Minutes elapsed: %f" % ((t1 - t0) / 60.))


	# with open(path+'logs/simple_lstm_v1_1.txt','wb') as handle:
	# 	pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)





if __name__ == "__main__":

	X_train,X_test,y_train,y_test = prepare_data()

	model = createModel()

	trainModel(model,X_train,X_test,y_train,y_test)


#536s