
if __name__ == '__main__':

	import pandas as pd
	import numpy as np
	from nltk.corpus import stopwords
	from scipy.spatial.distance import cosine 
	import re
	# import dask.dataframe as dd
	import cv2
	from multiprocessing import cpu_count,Pool


	from utils import *



	def w2v_sim(w1, w2):
		try: 
			return cosine(embeddings_index[w1], embeddings_index[w2])
		except KeyError:
			return 0.0

	def img_feature(row):
		s1 = row['question1']
		s2 = row['question2']
		
		t1 = re.split(r'\W',s1)
		t1 = filter(None, t1)
		t1 = [i for i in t1 if i not in stops]
		
		t2 = re.split(r'\W',s2)
		t2 = filter(None, t2)
		t2 = [i for i in t2 if i not in stops]  
	#     print t1, t2
		
		Z = [[w2v_sim(x, y) for x in t1] for y in t2] 
		a = np.array(Z, order='C')
	#     return [np.resize(a,(10,10)).flatten()]
		if np.sum(a) == 0:
			return np.zeros((10,10))
		return  [row.name, [cv2.resize(a ,(10,10)).flatten()]]





	stops = set(stopwords.words("english"))



	df_train = pd.read_csv('data/train_clean_vB.csv', 
						   dtype={
							   'q1_clean_vB': np.str,
							   'q2_clean_vB': np.str
						   })



	df_test = pd.read_csv('data/test_clean_vB.csv', 
						  dtype={
							  'q1_clean_vB': np.str,
							  'q2_clean_vB': np.str
						  })


	df_train.rename(columns={"q1_clean_vB":"question1", "q2_clean_vB":"question2"}, inplace=True)
	df_test.rename(columns={"q1_clean_vB":"question1", "q2_clean_vB":"question2"}, inplace=True)

	df_train['test_id'] = -1
	df_test['id'] = -1
	df_test['qid1'] = -1
	df_test['qid2'] = -1
	df_test['is_duplicate'] = -1

	df = pd.concat([df_train, df_test])
	df['question1'] = df['question1'].fillna('')
	df['question2'] = df['question2'].fillna('')
	df['uid'] = np.arange(df.shape[0])
	df = df.set_index(['uid'])


	ix_train = np.where(df['id'] >= 0)[0]
	ix_test = np.where(df['id'] == -1)[0]
	ix_is_dup = np.where(df['is_duplicate'] == 1)[0]
	ix_not_dup = np.where(df['is_duplicate'] == 0)[0]	

	df.fillna("empty",inplace=True)


	with open("data/cache/embeddings_index.npy", 'rb') as handle:
		embeddings_index = pickle.load(handle)

	print('Word embeddings: %d' % len(embeddings_index))





	def call_apply_fn(dft):
		return dft.apply(img_feature, axis=1)


	########### ACTION
	n_processes = 16
	pool = Pool(processes=n_processes)
	df_split = np.array_split(df, n_processes)

	print 'done init'



	pool_results = pool.map(call_apply_fn, df_split)
	new_df2 = pd.concat(pool_results)

	new_df2.to_csv("data/cache/pool_unusual_meaning_results.csv",index=False)

	print "finished running "


