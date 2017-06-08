''' 

getting rid of average layer
reducing network parameters


'''



from __future__ import print_function
import numpy as np
import pandas as pd
import datetime, time, json
import os.path
import pickle 

import keras
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

nr_hidden = 128
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
		print("TRAIN:", train_index, "TEST:", test_index)
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]

	if JUST_TESTING: #working with a subset of data for quick analysis
		X = X_train
		y = y_train

		sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=2021)
		for train_index, test_index in sss.split(X , y):
			X_train = X[train_index]
			y_train = y[train_index]

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

	#ToDo: projection
	Q.add(BatchNormalization())
	# Encode 
	# Q.add(Bidirectional(LSTM(nr_hidden, return_sequences=True,
	# 										 dropout=drop_out, recurrent_dropout=drop_out)))

	Q.add(LSTM(nr_hidden, return_sequences=True,
											 dropout=drop_out, recurrent_dropout=drop_out))



	# Q.add(TimeDistributed(Dense(nr_hidden, activation='relu', kernel_initializer= he_normal())))#, init='he_normal'
	# Q.add(TimeDistributed(BatchNormalization()))
	# Q.add(TimeDistributed(Dropout(drop_out)))

 
		
	
	return Q



def create_attention():

	attention_model = Sequential()
	# attention_model.add(Dropout(drop_out,input_shape=(nr_hidden,)))

	attention_model.add(Dense(nr_hidden, name='attend1',kernel_initializer= he_normal(),
				activation='relu',input_shape=(nr_hidden,)))
	attention_model.add(BatchNormalization())
	attention_model.add(Dropout(0.1))


	# attention_model.add(Dense(nr_hidden, name='attend2',kernel_initializer= he_normal(),
	# 			activation='relu'))
	# attention_model.add(BatchNormalization())

	attention_model = TimeDistributed(attention_model)

	return attention_model


def compare_sentence_and_alignment(question,alignment):
	comparison_model = Sequential()
	# comparison_model.add(Dropout(drop_out, input_shape=(nr_hidden*2,)))

	comparison_model.add(Dense(nr_hidden, name='compare1',kernel_initializer="he_normal",activation='relu', input_shape=(nr_hidden*2,)))
	comparison_model.add(BatchNormalization())
	comparison_model.add(Dropout(drop_out))

	# comparison_model.add(Dense(nr_hidden, name='compare2',kernel_initializer="he_normal",activation='relu'))
	# comparison_model.add(BatchNormalization())

	comparison_model = TimeDistributed(comparison_model)



	def get_features_by_comparison( sent, align):

		result = comparison_model(merge([sent, align], mode='concat')) # Shape: (i, n)
		avged = GlobalAveragePooling1D()(result)
		maxed = GlobalMaxPooling1D()(result)
		merged = merge([avged, maxed])
		result = BatchNormalization()(merged)
		result = BatchNormalization()(result)	
		return result


	# return get_features_by_comparison(question, alignment)
	return comparison_model(merge([question, alignment]))
def createModel():


	q1 = Input(shape=(MAX_SEQUENCE_LENGTH,))
	q2 = Input(shape=(MAX_SEQUENCE_LENGTH,))#,EMBEDDING_DIM

	encoder = create_bilstm()

	q1_encoded = encoder(q1)
	q2_encoded = encoder(q2)


	#Attention model


	attention_model = create_attention()
	attention1 = attention_model(q1_encoded)
	attention2 = attention_model(q2_encoded)


	def _outer(AB):
		att_ji = K.batch_dot(AB[1], K.permute_dimensions(AB[0], (0, 2, 1)))
		return K.permute_dimensions(att_ji,(0, 2, 1))

	co_attention = merge([attention1, attention2],
						 mode=_outer,
						 output_shape=(MAX_SEQUENCE_LENGTH, MAX_SEQUENCE_LENGTH))



	# Align

	def align(sentence, attention, transpose=False):
		def _normalize_attention(attmat):
			att = attmat[0]
			mat = attmat[1]
			if transpose:
				att = K.permute_dimensions(att,(0, 2, 1))
			# 3d softmax
			e = K.exp(att - K.max(att, axis=-1, keepdims=True))
			s = K.sum(e, axis=-1, keepdims=True)
			sm_att = e / s
			return K.batch_dot(sm_att, mat)
		
		return merge([attention, sentence], mode=_normalize_attention,
					  output_shape=(MAX_SEQUENCE_LENGTH,nr_hidden)) # Shape: (i, n)


	align1 = align(q2_encoded, co_attention) #alignment is normalized attention for the other question
	align2 = align(q1_encoded, co_attention, transpose=True)    


	# Compare


	feats1 = compare_sentence_and_alignment(q1_encoded, align1)
	feats2 = compare_sentence_and_alignment(q2_encoded, align2)	



	# Regression

	regressor_model = Sequential()

	# regressor_model.add(Dropout(drop_out, input_shape=(nr_hidden*2,)))
	regressor_model.add(Dense(200, name='entail1',kernel_initializer='he_normal',activation='relu',input_shape=(nr_hidden,)))
	regressor_model.add(BatchNormalization())
	regressor_model.add(Dropout(0.1))

	regressor_model.add(Dense(200, name='entail2', kernel_initializer='he_normal',activation='relu'))
	regressor_model.add(BatchNormalization())
	regressor_model.add(Dropout(0.1))

	# regressor_model.add(Dense(200, name='entail3', kernel_initializer='he_normal',activation='relu'))
	# regressor_model.add(Dropout(drop_out))
	# regressor_model.add(BatchNormalization())


	# regressor_model.add(Dense(200, name='entail4', kernel_initializer='he_normal',activation='relu'))
	# regressor_model.add(Dropout(drop_out))
	# regressor_model.add(BatchNormalization())

	regressor_model.add(Dense(1, name='entail_out', activation='sigmoid'))



	scores = regressor_model(merge([feats1, feats2], mode='concat'))
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


	MODEL_WEIGHTS_FILE = path+'weights/attention_v2_quick_epoch_{epoch:02d}_val_loss_{val_loss:.2f}.h5'

	#in case of broken training epochs
	# if os.path.isfile(path+"simple_bilstm_v1_00-0.46.h5"): 
	# 	model.load_weights(path+"simple_bilstm_v1_00-0.46.h5")

	early_stopping =EarlyStopping(monitor='val_loss', patience=5)
	callbacks = [ModelCheckpoint(MODEL_WEIGHTS_FILE, monitor='val_loss', save_best_only=True),early_stopping]


	print("Starting training at", datetime.datetime.now())
	t0 = time.time()




	history = model.fit([Q1_train, Q2_train],
						y_train,
						epochs=45,
						batch_size=256,
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