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

nr_hidden = 200
drop_out=0.2

JUST_TESTING = False
re_weight = True 



if re_weight:
	class_weight = {0: 1.309028344, 1: 0.472001959}
else:
	class_weight = None
	
	
weight_val = None





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

def ConvBlock(x, num_blocks, filters=32): 

    for i in range(num_blocks):
        x = Conv1D(filters, 4, activation='relu')(x)
        x = BatchNormalization()(x)        
        x = Dropout(0.1)(x)
        
    
    return x 

def create_graph():
	features = []

	
	graph_in = Input(shape=(MAX_SEQUENCE_LENGTH,)) #seems you can't miss this F*%!@$ layer
	
	
	embeddings = Embedding(nb_words + 1, 
				 EMBEDDING_DIM, 
				 weights=[word_embedding_matrix], 
				 input_length=MAX_SEQUENCE_LENGTH, 
				 trainable=False)(graph_in)

	init_batch = BatchNormalization()(embeddings)
	
	

	## use lstm output as a feature
	
	# v2
	lstm = LSTM(nr_hidden, return_sequences=False,dropout=0.2, recurrent_dropout=0.2)(init_batch)
	
	# lstm = GlobalMaxPooling1D()(lstm)
	# lstm = Dense(nr_hidden)(lstm)
	lstm = BatchNormalization()(lstm)	
	# lstm = Dropout(drop_out)(lstm)
	#v1
	# lstm = LSTM(nr_hidden, return_sequences=True,dropout=0.2, recurrent_dropout=0.2)(init_batch)

	features.append(lstm)
	
	
	# filter_length = 5
	# nb_filter = 64
	# pool_length = 4 

	#deep convs
	# x = ConvBlock(init_batch,3,32)
	# x = ConvBlock(x,2,64)
	# x = ConvBlock(x,2,128)
	x = ConvBlock(init_batch,2,64)
	x = GlobalMaxPooling1D()(x)

	x = Dense(nr_hidden)(x)
	x = BatchNormalization()(x)
	x = Dropout(drop_out)(x)


	# x = Flatten()(x)
	features.append(x)

	# put everything together
	out = keras.layers.concatenate(features) 

	# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> OUTPUT IS LSTM
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
	model = BatchNormalization()(model)
	model = Dropout(0.1)(model)

	model = Dense(200, activation='relu')(model)
	model = BatchNormalization()(model)
	model = Dropout(0.1)(model)

	# model = Dense(200, activation='relu')(model)
	# model = BatchNormalization()(model)
	# model = Dropout(0.1)(model)

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


	MODEL_WEIGHTS_FILE = path+'weights/lstm_and_shallow_convs_v2_epoch_{epoch:02d}_val_loss_{val_loss:.2f}.h5'

	#in case of broken training epochs
	# if os.path.isfile(path+"weights/simple_test_lstm_v1_epoch_13_val_loss_0.40.h5"): 
	# 	print('loading existing model weights')
	# 	model.load_weights(path+"weights/simple_test_lstm_v1_epoch_13_val_loss_0.40.h5")

	early_stopping =EarlyStopping(monitor='val_loss', patience=4)
	callbacks = [ModelCheckpoint(MODEL_WEIGHTS_FILE, monitor='val_loss', save_best_only=True),early_stopping]


	print("Starting training at", datetime.datetime.now())
	t0 = time.time()




	history = model.fit([Q1_train, Q2_train],
						y_train,
						epochs=50,
						batch_size=128,
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

	model = createModel()

	#getting weights set quickly first
	print ('getting quick weights with 50pct of data')
	print("")

	JUST_TESTING = True
	X_train,X_test,y_train,y_test = prepare_data()


	weight_val = np.ones(len(y_test))
	if re_weight:
		weight_val *= 0.472001959
		weight_val[y_test==0] = 1.309028344    



	trainModel(model,X_train,X_test,y_train,y_test)


	#fine tuning with whole dataset
	print("fine tuning with full dataset")
	print("")


	JUST_TESTING = False
	X_train,X_test,y_train,y_test = prepare_data()


	weight_val = np.ones(len(y_test))
	if re_weight:
		weight_val *= 0.472001959
		weight_val[y_test==0] = 1.309028344    

	trainModel(model,X_train,X_test,y_train,y_test)
