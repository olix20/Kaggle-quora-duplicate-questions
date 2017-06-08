import utils
from utils import *
import xgboost as xgb




train_features = load_array("data/cache/abhishek_qfreq_nostrov_lstm_neighbourcount_3gram_maxcore_train.dat")
test_features = load_array("data/cache/abhishek_qfreq_nostrov_lstm_neighbourcount_3gram_maxcore_test.dat")
labels = np.load(open(LABEL_TRAINING_DATA_FILE, 'rb'))


train_features = np.nan_to_num(train_features)
test_features = np.nan_to_num(test_features)


num_trials = 2
scores = []
models = []
params = []



def update_cache():
	global scores
	global models
	global params


	with open("data/cache/xgb_models_randomized.pik", 'w') as f:
	    pickle.dump(models,f,protocol=pickle.HIGHEST_PROTOCOL)
	    
	    
	with open("data/cache/xgb_scores_randomized.pik", 'w') as f:
	    pickle.dump(scores,f,protocol=pickle.HIGHEST_PROTOCOL)    
	    
	    
	with open("data/cache/xgb_params_randomized.pik", 'w') as f:
	    pickle.dump(params,f,protocol=pickle.HIGHEST_PROTOCOL)        



def read_cache():

	global scores
	global models
	global params



	with open("data/cache/xgb_models_randomized.pik", 'r') as f:
	    models = pickle.load(f)	    

	with open("data/cache/xgb_scores_randomized.pik", 'r') as f:
	    scores = pickle.load(f)	    

	with open("data/cache/xgb_params_randomized.pik", 'r') as f:
	    params = pickle.load(f)	    




read_cache()
print len(models), 'models so far'


for i in range(num_trials):
    perm = np.random.permutation(len(train_features))
    ix_fit = perm[:int(len(train_features)*(0.9))]
    ix_valid = perm[int(len(train_features)*(0.9)):]
    y_val = labels[ix_valid]
    y_train = labels[ix_fit]
    
    d_train = xgb.DMatrix(train_features[ix_fit],label= y_train)#, weight=weight_train)
    d_valid = xgb.DMatrix(train_features[ix_valid],label= y_val)#,weight=weight_val)

        
    xgb_params = {
    'max_depth': np.random.randint(6, 10), 
    'learning_rate': 0.01 + np.random.rand() * 0.02,
    'n_estimators': np.random.randint(2000, 4000), 
    'objective': 'binary:logistic',
        'eval_metric' :'logloss',
        'scale_pos_weight': 1,
                'gamma': np.random.rand() * 0.1, 
            'subsample': 0.75+np.random.rand() * 0.25, 
            'colsample_bytree': 0.75+np.random.rand() * 0.25, 
            'colsample_bylevel': 0.75+np.random.rand() * 0.25,
            'reg_alpha': np.random.rand() * 0.2, 
            'reg_lambda': 1.0 + np.random.rand() * 0.3, 


    }
    
    watchlist = [(d_train, 'train'), (d_valid, 'valid')]
    print xgb_params
    
    model = xgb.train(xgb_params, d_train, 10000,  watchlist, early_stopping_rounds=50, verbose_eval=100)#,feval=kappa)
    
    
    scores.append(model.best_score)
    models.append(model)
    params.append(xgb_params)
    
    print "best score and limit: ", model.best_score, model.best_ntree_limit
    
    update_cache()


