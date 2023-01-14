#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 23:56:38 2020

@author: seslay
"""

from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from skbio.stats.ordination import pcoa
from skbio.stats.distance import DistanceMatrix
from scipy.spatial.distance import pdist, squareform
import pandas as pd
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import NearestCentroid
from collections import Counter
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle
import seaborn as sns
from sklearn.preprocessing import normalize


import numpy as np
import pandas as pd
import warnings
import csv
from itertools import groupby

from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.semi_supervised import LabelPropagation
from sklearn.semi_supervised import LabelSpreading
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import RidgeClassifierCV
from sklearn.neighbors import NearestCentroid
from sklearn.linear_model import RidgeClassifier

from sklearn.metrics import recall_score as rec, precision_score as pre, f1_score as f1, accuracy_score as acc

clf_dict={}
#clf_dict['knn'] = KNeighborsClassifier(weights='distance')
clf_dict['lr'] = LogisticRegression(random_state=0, max_iter=400, multi_class='multinomial', solver='newton-cg')
clf_dict['svm'] = make_pipeline(StandardScaler(), LinearSVC(random_state=0, multi_class='crammer_singer', tol=1e-9, max_iter=2000))
clf_dict['mlp_sklearn'] = MLPClassifier(random_state=1, max_iter=400, early_stopping=False, )
clf_dict['rand_for'] = RandomForestClassifier(max_depth=4, criterion='entropy')
clf_dict['dec_tree'] = DecisionTreeClassifier(random_state=0, criterion='entropy')
clf_dict['ber_nb'] = BernoulliNB()
clf_dict['xtra_tree'] = ExtraTreesClassifier(n_estimators=150, criterion='entropy',max_depth = 25)
clf_dict['gau_nb'] = GaussianNB()
clf_dict['label_prop'] = LabelPropagation(kernel='knn')
clf_dict['label_spread'] = LabelSpreading(kernel='knn')
clf_dict['lda'] = LinearDiscriminantAnalysis()
clf_dict['ridge_cv'] = RidgeClassifierCV()
clf_dict['n_cent'] = NearestCentroid()
clf_dict['ridge'] = RidgeClassifier()

















nrps_file = pd.read_csv("/Users/seslay/Downloads/match_index/KS_alignment/data_Adomain_Substrate_labeled_sigs.report",sep = '\t')
nrps_file = nrps_file.drop_duplicates('sig')

nrps_signature = nrps_file['sig']
nrps_label = nrps_file['sub']
nrps_name = list(nrps_file['#sequence-id'])

                           
                           
                           
                    
                           
                           
                           
                           
                           
def get_correct_sub(sub_in):
    if sub_in.lower() == 'beta-ala':
        ret = 'ala'
    elif sub_in.lower() == 'orn':
        ret = 'orn'
    elif sub_in.lower() == 'hyv-d':
        ret = 'null'
    else:
        ret = sub_in
    
    return ret

nrps_label = [get_correct_sub(i) for i in nrps_label]
z_scores_df = pd.read_csv('/Users/seslay/Downloads/score_aa_nrpspredictor2.csv', delimiter=',') 
score_matrix = z_scores_df.drop(['amino_acid_1', 'amino_acid_3'], axis=1)

aa_dict = {}

amino_acid_order = list(z_scores_df["amino_acid_1"])
for item in amino_acid_order:
    aa_dict[item] = score_matrix.loc[amino_acid_order.index(item),:].to_numpy()



aa_dict['-'] = list(np.mean(np.asarray(list(aa_dict.values())), axis=0))
aa_dict['B'] = list(np.mean(np.asarray(list([aa_dict['D'],aa_dict['N']])), axis=0))


def get_name_signature_nrps(string_label,string_signature):    
    label = []
    label = [each_string.lower() for each_string in string_label]
    index_list = list(set(label))
    label_index = [index_list.index(this_item) for this_item in label]
    X_matrix = []
    
    for str_this in string_signature:
        print(str_this)
        X_matrix.append(get_encoding(str_this))
    return label_index,X_matrix,index_list

def get_encoding(signature):
    ret = []
    for i in signature:
        ret.extend(aa_dict[i])
    return np.asarray(ret)

nrps_label_index,nrps_X_matrix,nrps_index_list = get_name_signature_nrps(nrps_label,nrps_signature)

def get_nrps_accuracy(nrps_label_index,nrps_X_matrix,nrps_index_list,clf_list):
    clf_score = clf_list.copy()
    for item in clf_score:
        clf_score[item] = []
    seed = [0,1,2,3,4,5,6,7,8,9]
    slow_iter = 0
    score = []
    for k in range(10):
        block = int(len(nrps_X_matrix)/5)
        print(k)
        for i in range(5):
            shuffled_index_for_ml = shuffle(range(len(nrps_X_matrix)),random_state=seed[k])
            data_ML = np.array([nrps_X_matrix[i] for i in shuffled_index_for_ml])
            label_ML = np.array([nrps_label_index[i] for i in shuffled_index_for_ml])
            print(len(data_ML))
            train_data_ML = data_ML[np.r_[0:i*block,(i+1)*block:len(data_ML)],:]
            test_data_ML = data_ML[i*block:(i+1)*block,:]
            label_train_ML = np.array(label_ML)[np.r_[0:i*block,(i+1)*block:len(label_ML)]]
            label_test_ML = np.array(label_ML)[i*block:(i+1)*block]
            if i == 4:
                test_data_ML = data_ML[i*block:len(data_ML),:]
                label_test_ML = label_ML[i*block:len(data_ML)]
            for item in clf_list:
                if item == 'svm' or item == 'lr':
                    slow_iter = slow_iter + 1
                    if slow_iter>10:
                        continue                    
                    
                clf = clf_list[item]
                clf.fit(train_data_ML, label_train_ML)
                this_score = clf.score(test_data_ML,label_test_ML)
                clf_score[item].append(this_score)
    
    for item in clf_score:
        clf_score[item] = np.array(clf_score[item]).mean()
    return clf_score

nrps_score_origin = get_nrps_accuracy(nrps_label_index,nrps_X_matrix,nrps_index_list,clf_dict)

output_dict = pd.DataFrame.from_dict(nrps_score_origin, orient="index")
output_dict.to_csv('/Users/seslay/Downloads/match_index/KS_alignment/NRPS_result/NRPS_origin_result.csv',sep = ',',index = True)


def adding_3D_info(input_path,X_matrix,string_name,label_index):
    X_matrix_this = X_matrix.copy()
    label_index_this = label_index.copy()
    string_name_this =string_name.copy()
    D_info = open(input_path,"r")
    max_len = 0
    for line in D_info:
        name = line.split("\t")[0]
        scores = line.rstrip().split("\t")[1:]
        if name not in string_name:
            continue
        this_index = string_name.index(name)

        print(np.append(X_matrix[this_index],np.array(scores).astype(float)).shape)
        X_matrix_this[this_index] = np.append(X_matrix_this[this_index],np.array(scores).astype(float))
        max_len = len(X_matrix_this[this_index])
    deleted_item = []
    for k in range(len(X_matrix_this)):
        if len(X_matrix_this[k])<max_len:
            deleted_item.append(k)
    X_matrix_this = np.delete(X_matrix_this, deleted_item, 0)
    label_index_this = [label_index_this[m] for m in range(len(label_index_this)) if m not in deleted_item]
    string_name_this = [string_name_this[m] for m in range(len(string_name_this)) if m not in deleted_item]
        #print(X_matrix_this[this_index])
    return X_matrix_this,string_name_this,label_index_this
    
#buck = []

nrps_X_matrix_3D_info,nrps_name,nrps_label_index_3D_info = adding_3D_info("/Users/seslay/Downloads/match_index/KS_alignment/3D_structure/3D_info_summary_nrps_"+str("ss8")+".txt",nrps_X_matrix,nrps_name,nrps_label_index)
print(len(nrps_X_matrix_3D_info[0]))
nrps_X_matrix_3D_info,nrps_name,nrps_label_index_3D_info = adding_3D_info("/Users/seslay/Downloads/match_index/KS_alignment/3D_structure/3D_info_summary_nrps_"+str("ss3")+".txt",nrps_X_matrix_3D_info,nrps_name,nrps_label_index_3D_info)
nrps_X_matrix_3D_info,nrps_name,nrps_label_index_3D_info = adding_3D_info("/Users/seslay/Downloads/match_index/KS_alignment/3D_structure/3D_info_summary_nrps_"+str("acc")+".txt",nrps_X_matrix_3D_info,nrps_name,nrps_label_index_3D_info)
nrps_X_matrix_3D_info,nrps_name,nrps_label_index_3D_info = adding_3D_info("/Users/seslay/Downloads/match_index/KS_alignment/3D_structure/3D_info_summary_nrps_"+str("diso")+".txt",nrps_X_matrix_3D_info,nrps_name,nrps_label_index_3D_info)


nrps_score_3D_info = get_nrps_accuracy(nrps_label_index_3D_info,nrps_X_matrix_3D_info,nrps_index_list,clf_dict)


output_dict_3D = pd.DataFrame.from_dict(nrps_score_3D_info, orient="index")
output_dict_3D.to_csv('/Users/seslay/Downloads/match_index/KS_alignment/NRPS_result/NRPS_RaptorX_result.csv',sep = ',',index = True)













