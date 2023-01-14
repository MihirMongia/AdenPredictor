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
from sklearn.metrics import confusion_matrix







nrps_file = pd.read_csv("/Users/seslay/Downloads/match_index/KS_alignment/data_Adomain_Substrate_labeled_sigs.report",sep = '\t')
nrps_file = nrps_file.drop_duplicates('sig')
len(set(nrps_label))
nrps_signature = list(nrps_file['sig'])
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
       # print(str_this)
        X_matrix.append(get_encoding(str_this))
    return label_index,X_matrix,index_list

def get_encoding(signature):
    ret = []
    for i in signature:
        ret.extend(aa_dict[i])
    return np.asarray(ret)

nrps_label_index,nrps_X_matrix,nrps_index_list = get_name_signature_nrps(nrps_label,nrps_signature)

def get_nrps_accuracy(nrps_label_index,nrps_signature,nrps_index_list):
    seed = [0,1,2,3,4,5,6,7,8,9]
    score = []
    correct_label_list = []
    predicted_label_list = []
    for k in range(10):
        block = int(len(nrps_signature)/5)
        print(k)
        for i in range(5):
            correct_label_count = 0
            shuffled_index_for_ml = shuffle(range(len(nrps_signature)),random_state=seed[k])
            data_ML = np.array([nrps_signature[sm] for sm in shuffled_index_for_ml])
            label_ML = np.array([nrps_label_index[sm] for sm in shuffled_index_for_ml])
            print(len(data_ML))
            train_data_ML = data_ML[np.r_[0:i*block,(i+1)*block:len(data_ML)]]
            test_data_ML = data_ML[i*block:(i+1)*block]
            label_train_ML = label_ML[np.r_[0:i*block,(i+1)*block:len(label_ML)]]
            label_test_ML = label_ML[i*block:(i+1)*block]
            if i == 4:
                test_data_ML = data_ML[i*block:len(data_ML)]
                label_test_ML = label_ML[i*block:len(data_ML)]
            label_possibility_list = {}
            sig_score_list = {}
            for m in range(len(nrps_index_list)):
                training_data_this_label = [train_data_ML[ts] for ts in range(len(train_data_ML)) if label_train_ML[ts]==m]
                label_possibility_list[m] = len(training_data_this_label)/len(train_data_ML)
                sig_score_list[m] = {}
                for pos in range(34):
                    sig_score_list[m][pos] = {}
                    for dist_sig in aa_dict:
                        this_sig_count = 0
                        
                        for ie in range(len(training_data_this_label)):
                            this_sig_count = this_sig_count+1  if training_data_this_label[ie][pos] == dist_sig else this_sig_count
                        sig_score_list[m][pos][dist_sig] = 0.001 if this_sig_count == 0 else this_sig_count/len(training_data_this_label)
            for tst in range(len(test_data_ML)):
                predicted_possibility = []
                tst_signature = test_data_ML[tst]
                tst_label = label_test_ML[tst]
                for lbl in label_possibility_list:
                    this_pos = label_possibility_list[lbl]
                    for tst_pos in range(34):
                        this_pos = this_pos*sig_score_list[lbl][tst_pos][tst_signature[tst_pos]]
                    predicted_possibility.append(this_pos)
                predicted_label = predicted_possibility.index(max(predicted_possibility))
                if predicted_label == tst_label:
                    correct_label_count+=1
                else:
                    print(tst_signature)
                correct_label_list.append(tst_label)
                predicted_label_list.append(predicted_label)
            score.append(correct_label_count/len(test_data_ML))
            
    return np.array(score).mean(),correct_label_list,predicted_label_list

nrps_score_possibility,corre_list,pred_list = get_nrps_accuracy(nrps_label_index,nrps_signature,nrps_index_list)



confusion_df = pd.DataFrame(confusion_matrix(corre_list, pred_list,labels=list(set(nrps_label_index))))
confusion_df.columns = nrps_index_list
confusion_df.index = nrps_index_list

confusion_df = confusion_df/10
sns.set(font_scale=1.5)

fig, ax = plt.subplots(1, 1, figsize=(16, 10), dpi= 200)

plt.subplots_adjust(hspace = 0.8)
ax = sns.heatmap(np.clip(confusion_df, 0, 15),cmap = "Blues",xticklabels=nrps_index_list, yticklabels=nrps_index_list,cbar_kws={"shrink": 0.5})
plt.subplots_adjust(bottom=0.22)
plt.subplots_adjust(left=0.15)
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')


fig.savefig('/Users/seslay/Downloads/match_index/KS_alignment/nrps_prob_heatmap.png', format='png')















