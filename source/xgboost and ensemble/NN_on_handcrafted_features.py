
import utils
from utils import *

import keras
from keras import backend as K
from keras.utils.data_utils import get_file
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Input, Embedding, Reshape, merge, LSTM, Bidirectional
from keras.layers import TimeDistributed, Activation, SimpleRNN, GRU
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, RMSprop, Adam
from keras.metrics import categorical_crossentropy, categorical_accuracy
from keras.layers.convolutional import *
from keras.preprocessing import image, sequence
from keras.preprocessing.text import Tokenizer
from sklearn.metrics import fbeta_score
from keras.callbacks import ReduceLROnPlateau, EarlyStopping,ModelCheckpoint



train_features = load_array("data/cache/all_features_v3_triangle_train.dat")
test_features = load_array("data/cache/all_features_v3_triangle_test.dat")


train_features = np.nan_to_num(train_features)
test_features = np.nan_to_num(test_features)
labels = np.load(open(LABEL_TRAINING_DATA_FILE, 'rb'))



from sklearn.preprocessing import StandardScaler



num_folds = 5
X = train_features
X_submission = np.nan_to_num(StandardScaler().fit_transform(test_features))

y = labels



np.random.seed(2189)

perm = np.random.permutation(len(train_features))
ix_fit = perm[:int(len(train_features)*(1-0.1))]
ix_valid = perm[int(len(train_features)*(1-0.1)):]

X_train = np.nan_to_num(StandardScaler().fit_transform(X[ix_fit]))
X_valid = np.nan_to_num(StandardScaler().fit_transform(X[ix_valid]))


y_val = labels[ix_valid]
y_train = labels[ix_fit]



def NN():
	nf=64; p=0.5
	
	num_hidden1 = np.random.randint(512, 1024)
	num_hidden2 = np.random.randint(512, 1024)
	rate_drop_1 = 0.3 + np.random.rand() * 0.25
	rate_drop_2 = 0.3 + np.random.rand() * 0.25

	STAMP = 'NN_%d_%d_%.2f_%.2f'%(num_hidden1, num_hidden2, rate_drop_1, \
			rate_drop_2)


	print STAMP


	layers =  [
		Dense(num_hidden1,activation='relu',input_shape =  (train_features.shape[1],)),
		Dropout(rate_drop_1),   
		BatchNormalization(),


		Dense(num_hidden2,activation='relu'),
		Dropout(rate_drop_2),   
		BatchNormalization(),


		Dense(1,activation='sigmoid')]
	
	
	return  Sequential(layers), STAMP



class_weight = {0: 1.309028344, 1: 0.472001959}


weight_val = np.ones(len(y_val))
weight_val *= 0.472001959
weight_val[y_val==0] = 1.309028344




val_preds = np.zeros((y_val.shape[0],1))
test_preds = np.zeros((test_features.shape[0],1))
val_scores = []



for i in range(10):

	model,STAMP =NN()


	kfold_weights_path = "weights/NN_features_{}.h5".format(STAMP)
	callbacks = [EarlyStopping(monitor='val_loss', patience=4),
				 ModelCheckpoint(kfold_weights_path, monitor='val_loss', save_best_only=True)]


	model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])


	history = model.fit(x = X_train, y= y_train, validation_data=(X_valid, y_val,weight_val),
	  batch_size=128, epochs=100,callbacks=callbacks,  class_weight=class_weight, shuffle=True,verbose=1)



	bst_val_score = min(history.history['val_loss'])

	print ("bst_val_score: ", bst_val_score )

	val_scores.append(bst_val_score)



	print('Start making the submission before fine-tuning')
	model.load_weights(kfold_weights_path)


	val_preds += model.predict(X_valid, batch_size=2048, verbose=1).reshape(-1,1)

	test_preds += model.predict(X_submission, batch_size=2048, verbose=1).reshape(-1,1)
	# preds += model.predict([test_data_2, test_data_1,abhishek_np_test], batch_size=8192, verbose=1)
   




save_array("data/results/val_preds.dat",val_preds)
save_array("data/results/test_preds.dat",test_preds)
save_array("data/results/scores.dat",val_scores)
