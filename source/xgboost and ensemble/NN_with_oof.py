
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


from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


train_features = load_array("data/cache/train_features_extrasvd_v2.dat")
test_features = load_array("data/cache/test_features_extrasvd_v2.dat" )
# col_labels = load_array("data/cache/col_labels_extrasv2v2.dat")

# train_features = np.nan_to_num(train_features)
# test_features = np.nan_to_num(test_features)
labels = np.load(open(LABEL_TRAINING_DATA_FILE, 'rb'))






num_folds = 5
X = train_features
X_submission = np.nan_to_num(StandardScaler().fit_transform(test_features))

y = labels



# np.random.seed(2189)

# perm = np.random.permutation(len(train_features))
# ix_fit = perm[:int(len(train_features)*(1-0.1))]
# ix_valid = perm[int(len(train_features)*(1-0.1)):]

# X_train = np.nan_to_num(StandardScaler().fit_transform(X[ix_fit]))
# X_valid = np.nan_to_num(StandardScaler().fit_transform(X[ix_valid]))


# y_val = labels[ix_valid]
# y_train = labels[ix_fit]



def NN(num_hidden1,num_hidden2,rate_drop_1,rate_drop_2):
	# p=0.5
	

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



# class_weight = {0: 1.309028344, 1: 0.472001959}


# weight_val = np.ones(len(y_val))
# weight_val *= 0.472001959
# weight_val[y_val==0] = 1.309028344




y_oof = np.zeros((train_features.shape[0],1))
test_preds = np.zeros((test_features.shape[0],1))
val_scores = []


num_iters = 5
for i in range(num_iters):


	## create random params for current iteration
	num_hidden1 = np.random.randint(512, 1024)
	num_hidden2 = np.random.randint(512, 1024)
	rate_drop_1 = 0.3 + np.random.rand() * 0.25
	rate_drop_2 = 0.3 + np.random.rand() * 0.25



	num_folds = 5
	current_fold = 0 
	skf = StratifiedKFold(n_splits=num_folds, random_state=2019)
	splits = skf.split(train_features, labels)

	for ix_fit, ix_valid in tqdm(splits, total=num_folds):
		

		y_val = labels[ix_valid]
		y_train = labels[ix_fit]

		X_train = np.nan_to_num(StandardScaler().fit_transform(train_features[ix_fit]))
		X_valid = np.nan_to_num(StandardScaler().fit_transform(train_features[ix_valid]))



		current_fold +=1

		
		##### create model
		model,STAMP =NN(num_hidden1,num_hidden2,rate_drop_1,rate_drop_2)


		kfold_weights_path = "weights/NN_oof_{}_fold_{}.h5".format(STAMP,current_fold)
		callbacks = [EarlyStopping(monitor='val_loss', patience=4),
					 ModelCheckpoint(kfold_weights_path, monitor='val_loss', save_best_only=True)]


		model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])



		history = model.fit(x = X_train, y= y_train, validation_data=(X_valid, y_val),
		  batch_size=128, epochs=100,callbacks=callbacks,  shuffle=True,verbose=1)


		########### evaluation

		bst_val_score = min(history.history['val_loss'])
		tqdm.write("bst_val_score:{}".format(bst_val_score) )
		val_scores.append(bst_val_score)



		tqdm.write('Start making the submission before fine-tuning')
		model.load_weights(kfold_weights_path)


		y_oof[ix_valid] += model.predict(X_valid, batch_size=8192, verbose=1).reshape(-1,1)

		test_preds += model.predict(X_submission, batch_size=8192, verbose=1).reshape(-1,1)
		# preds += model.predict([test_data_2, test_data_1,abhishek_np_test], batch_size=8192, verbose=1)
	   




save_array("data/results/NN_5fold_s2019_oof_preds_extrasvdv2.dat",(y_oof/float(num_iters)))
save_array("data/results/NN_5fold_s2019_test_preds_extrasvdv2.dat",(test_preds/float(num_iters)))
save_array("data/results/NN_val_scores.dat",val_scores)
