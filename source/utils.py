# from __future__ import print_function

import bcolz
import pickle 

import numpy as np
import pandas as pd
import csv, json
from zipfile import ZipFile
from os.path import expanduser, exists
from IPython.lib.display import FileLink



import sys
import os 
import gc
from tqdm import tqdm, tqdm_notebook



# import spacy


import matplotlib.pyplot as plt
import seaborn as sns

import datetime, time, json



#from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, make_scorer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import StratifiedKFold 








########### Constants

path = '/home/ubuntu/quora/'
data_home = path +"data/"


KERAS_DATASETS_DIR = data_home+"cache/"

Q1_TRAINING_DATA_FILE = data_home+'cache/q1_train.npy'
Q2_TRAINING_DATA_FILE = data_home+'cache/q2_train.npy'
LABEL_TRAINING_DATA_FILE = data_home+'cache/label_train.npy'
WORD_EMBEDDING_MATRIX_FILE = data_home+'cache/word_embedding_matrix.npy'
NB_WORDS_DATA_FILE = data_home+'cache/nb_words.json'
Q1_TESTING_DATA_FILE = data_home+'q1_test.npy'
Q2_TESTING_DATA_FILE = data_home+'q2_test.npy'


MAX_NB_WORDS = 200000
MAX_SEQUENCE_LENGTH = 35#25
EMBEDDING_DIM = 300



#####  RE-weighting ref:https://github.com/0celot/mlworkshop39_042017/blob/master/3_masterclass/ipy/feature_extraction.ipynb
re_weight = True # whether to re-weight classes to fit the 17.5% share in test set

score0 = 6.01888
score1 = 28.52056
score05 = 0.69315
score025 = 0.47913
score075 = 1.19485

A = (score0 + score1)*np.log(0.5)/score05
eps = 10e-16
B = np.log(1 - eps)
C = np.log(eps)
r1 = (score1 - (C/B)*score0) / ((C*C/B) - B)
r0 = (-score1 - r1*B)/C

gamma_0 = 1.30905513329
gamma_1 = 0.472008228977


def link_function(x):
    return gamma_1*x/(gamma_1*x + gamma_0*(1 - x))




########### Functions

def save_array(fname, arr):
    c=bcolz.carray(arr, rootdir=fname, mode='w')
    c.flush()


def load_array(fname):
    return bcolz.open(fname)[:]


def log_loss_lf(y_true, y_pred):
    return log_loss(y_true, link_function(y_pred[:, 1]), eps=eps)


def log_loss_lf_xgb(y_pred, y_true):
        return 'llf', log_loss(y_true.get_label(), link_function(y_pred), eps=eps)


    
    
####  plot feature split by class    


def plot_real_feature(df, fname):
    plt.rcParams["figure.figsize"] = (15,10)
    
    
    fig = plt.figure()
    ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2)
    ax2 = plt.subplot2grid((3, 2), (1, 0), colspan=2)
    ax3 = plt.subplot2grid((3, 2), (2, 0))
    ax4 = plt.subplot2grid((3, 2), (2, 1))
    ax1.set_title('Distribution of %s' % fname, fontsize=20)
    sns.distplot(df.loc[ix_train][fname], 
                 bins=50, 
                 ax=ax1)    
    sns.distplot(df.loc[ix_is_dup][fname], 
                 bins=50, 
                 ax=ax2,
                 label='is dup')    
    sns.distplot(df.loc[ix_not_dup][fname], 
                 bins=50, 
                 ax=ax2,
                 label='not dup')
    ax2.legend(loc='upper right', prop={'size': 18})
    sns.boxplot(y=fname, 
                x='is_duplicate', 
                data=df.loc[ix_train], 
                ax=ax3)
    sns.violinplot(y=fname, 
                   x='is_duplicate', 
                   data=df.loc[ix_train], 
                   ax=ax4)
    plt.show()
    
    
    