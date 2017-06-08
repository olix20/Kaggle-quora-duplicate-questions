import pandas as pd
import numpy as np 
import bcolz
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis



def save_array(fname, arr):
    c=bcolz.carray(arr, rootdir=fname, mode='w')
    c.flush()
    
    
def load_array(fname):
    return bcolz.open(fname)[:]
    





def create_and_save_features(features, path):
	question1_vectors_train  = features[:,:175]
	question2_vectors_train = features[:,175:]

	q12_combined_train = zip(question1_vectors_train,question2_vectors_train)

	data = {}

	print ('creating cosine_distance')
	data['cosine_distance'] = [cosine(x, y) for (x, y) in q12_combined_train]

	print ('creating cityblock_distance')

	data['cityblock_distance'] = [cityblock(x, y) for (x, y) in q12_combined_train]

	print ('creating jaccard_distance')

	data['jaccard_distance'] = [jaccard(x, y) for (x, y) in q12_combined_train]

	print ('creating canberra_distance')

	data['canberra_distance'] = [canberra(x, y) for (x, y) in q12_combined_train]

	print ('creating euclidean_distance')

	data['euclidean_distance'] = [euclidean(x, y) for (x, y) in q12_combined_train]

	print ('creating minkowski_distance')

	data['minkowski_distance'] = [minkowski(x, y, 3) for (x, y) in q12_combined_train]

	print ('creating braycurtis_distance')

	data['braycurtis_distance'] = [braycurtis(x, y) for (x, y) in q12_combined_train]


	lstm_features_train_q1_q2 = pd.DataFrame.from_dict(data)
	lstm_features_train_q1_q2.to_csv(path,index=False)




### Loaad data

lstm_features_train_q12 = load_array("data/cache/lstm_features_train_q1_q2.dat/")
lstm_features_test_q12 = load_array("data/cache/lstm_features_test_q1_q2.dat/")




create_and_save_features(lstm_features_train_q12,"data/cache/lstm_distances_train_q12.csv")
create_and_save_features(lstm_features_test_q12,"data/cache/lstm_distances_test_q12.csv")

