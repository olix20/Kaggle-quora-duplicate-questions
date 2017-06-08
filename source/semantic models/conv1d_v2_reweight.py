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



'''

results: poor, ran for 25+ epochs with val accuracy standing around 0.6
best val_acc: 0.44
tried with resnet
'''




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


	return X_train,X_test,y_train,y_test



def get_convs():
#v2
	graph_in = Input(shape=(MAX_SEQUENCE_LENGTH,EMBEDDING_DIM))
	normalized_input = BatchNormalization()(graph_in)
	
	convs = []

	for fsz in range(2,6):
		conv = Conv1D(16, fsz,
					  padding = 'valid', activation='relu')(normalized_input)#
		conv = BatchNormalization()(conv)        
		conv = Dropout(0.1)(conv)
		
		pool = GlobalMaxPooling1D()(conv)
		convs.append(pool)

	out = Merge(mode='concat')(convs) 
	graph = Model(input=graph_in,output=out)

	return graph

	

def createModel():


	graph = get_convs()

	Q1 = Sequential()
	Q1.add(Embedding(nb_words + 1, 
					 EMBEDDING_DIM, 
					 weights=[word_embedding_matrix], 
					 input_length=MAX_SEQUENCE_LENGTH, 
					 trainable=False))


	Q1.add(graph)



	### Same ops for question 2

	Q2 = Sequential()
	Q2.add(Embedding(nb_words + 1, 
					 EMBEDDING_DIM, 
					 weights=[word_embedding_matrix], 
					 input_length=MAX_SEQUENCE_LENGTH, 
					 trainable=False))


	Q2.add(graph)


	model = Sequential()
	model.add(Merge([Q1, Q2], mode='concat'))
	# concat = Concatenate([Q1, Q2])
	# model.add(concat)

	model.add(BatchNormalization())
	model.add(Dense(200, activation='relu'))
	model.add(BatchNormalization())

	model.add(Dense(200, activation='relu'))
	model.add(BatchNormalization())

	model.add(Dense(1, activation='sigmoid'))

	return model




def trainModel(model,X_train,X_test,y_train,y_test):

	Q1_train = X_train[:,0]
	Q2_train = X_train[:,1]
	Q1_test = X_test[:,0]
	Q2_test = X_test[:,1]





	model.compile(loss='binary_crossentropy', 
			  optimizer="nadam", 
			  metrics=['accuracy'])#, 'precision', 'recall', 'fbeta_score'])

	


	MODEL_WEIGHTS_FILE = path+'weights/conv1d_v2_reweight_epoch_{epoch:02d}_val_loss_{val_loss:.2f}.h5'



	early_stopping =EarlyStopping(monitor='val_loss', patience=5)
	callbacks = [ModelCheckpoint(MODEL_WEIGHTS_FILE, monitor='val_loss', save_best_only=True),early_stopping]


	print("Starting training at", datetime.datetime.now())
	t0 = time.time()



	history = model.fit([Q1_train, Q2_train],
						y_train,
						epochs=50,
						batch_size=128,
						validation_data = ([Q1_test, Q2_test],y_test,weight_val),
						class_weight=class_weight,
						callbacks=callbacks,shuffle=True)





	t1 = time.time()
	print("Training ended at", datetime.datetime.now())
	print("Minutes elapsed: %f" % ((t1 - t0) / 60.))



if __name__ == "__main__":

	X_train,X_test,y_train,y_test = prepare_data()
	weight_val = np.ones(len(y_test))
	
	if re_weight:
		weight_val *= 0.472001959
		weight_val[y_test==0] = 1.309028344    




	model = createModel()
	trainModel(model,X_train,X_test,y_train,y_test)


#536s