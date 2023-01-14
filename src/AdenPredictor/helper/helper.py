#!/usr/bin/env python
# coding: utf-8

import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import csv
import sys
import getopt
from datetime import datetime
from collections import Counter

# from sklearn.svm import LinearSVC
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
# from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import ExtraTreesClassifier
# from sklearn.tree import ExtraTreeClassifier as ExtraTreesClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.semi_supervised import LabelPropagation
# from sklearn.semi_supervised import LabelSpreading
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.linear_model import RidgeClassifierCV
# from sklearn.neighbors import NearestCentroid
# from sklearn.linear_model import RidgeClassifier

from sklearn.metrics import recall_score as rec, precision_score as pre, f1_score as f1, accuracy_score as acc

def remove_multiple_spaces(string, remove_lead_trail=True):
    if remove_lead_trail:
        string = string.strip()
    while '  ' in string:
        string = string.replace('  ', ' ')
    return string

def get_state(state, line, print_state_transitions=False):
    #print(line)
    #print('incoming state ', state)
    newstate = ''
    if state in ['init', 'parsing']:
        if line.startswith('//') or line.startswith('- - - -'):
            newstate = 'recognise'
    elif state == 'recognise':
        if line.startswith('Alignments of top'):
            newstate = 'parsing'

    if print_state_transitions:
        if newstate != '':
            print('Changed state from ',state,' to ', newstate)
        else:
            newstate = state
    
    if newstate == '':
        newstate = state
    return newstate

def parse_hmmsearch_output(lines, hmmfiles):
    align_dict = {}
    align_code_dict = {}
    score_dict = {}
    detail_dict = {}
    line_score_idx = 0
    
    
    line_idx = 0
    line_align_idx = 0
    Id = ''
    state = 'init'
    curr_hmm = ''
    #print(lines)
    
    for line in lines:
        line_idx += 1        
        state = get_state(state, line)
        
        if state == 'recognise':
            if line.startswith('Query sequence'):
                Id = line.split(': ')[1].strip('\n')
                #print('set id to ',Id)
                detail_dict[Id] = {}

        elif state == 'parsing':
            hmmheader = np.asarray([line.startswith(hmmfile) for hmmfile in hmmfiles]).any()
            if hmmheader:
                line_align_idx = line_idx
                curr_hmm = line.split(':')[0]
                #print(curr_hmm)
                split = line.split(' ')
                #print(split)
                score_idx = split.index('score')+1
                from_idx = split.index('from')+1
                to_idx = split.index('to')+1
                detail_dict[Id][curr_hmm] = {'score':float(split[score_idx].strip(',')),'from': int(split[from_idx]),
                                             'to': int(split[to_idx].strip(':')), 'top':'', 'bottom':''}
                #print(detail_dict)
            elif line.startswith(' '):
                #print(line_idx, line_align_idx, line)
                if (line_idx - line_align_idx) % 4 == 1:
                    detail_dict[Id][curr_hmm]['top'] += remove_multiple_spaces(line).strip('*-><')
                elif (line_idx - line_align_idx) % 4 == 3:
                    detail_dict[Id][curr_hmm]['bottom'] += remove_multiple_spaces(line).split()[2]
        
    return detail_dict

def parse_hmmsearch_output_from_file(filename, hmmfile):
    with open(filename, 'r') as file:
        content = file.readlines()
    return parse_hmmsearch_output(content, hmmfile)

def get_hmm_alignment(mydict, hmm):
    ret_dict = {}
    for key in mydict:
        try:
            ret_dict[key] = mydict[key][hmm].copy()
        except:
            print('Could not get',hmm,' for Id',key)
    return ret_dict

def removetopindels(indict, print_change=False):
    mydict = indict.copy()
    for Id in mydict:
        top_tmp = ''
        bot_tmp = ''
        idx = mydict[Id]['from']
        idx_list = []
        for a,b in zip(mydict[Id]['top'], mydict[Id]['bottom']):
            if a != '.':
                top_tmp += a
                bot_tmp += b
                if b == '-':
                    idx_list.append(idx-0.5)
                else:
                    idx_list.append(idx)
            if b != '-':
                idx += 1
        if print_change and mydict[Id]['top'] != top_tmp:
            print('Id:',Id,' top changed from ',mydict[Id]['top'], 'to', top_tmp)
        if print_change and mydict[Id]['bottom'] != bot_tmp:
            print('Id:',Id,' bottom changed from ',mydict[Id]['bottom'], 'to', bot_tmp)
        mydict[Id]['top'] = top_tmp
        mydict[Id]['bottom'] = bot_tmp
        assert(len(mydict[Id]['bottom']) == len(idx_list))
        mydict[Id]['idx_list'] = idx_list.copy()
    return mydict

def extractCharacters(Id, target, source, source_idx_list, pattern, idxs) :
    assert len(source) == len(source_idx_list)
    try:
        start = target.index(pattern)
    except:
        print('Problem at Id ', Id, ' pattern ', pattern, ' target ', target)
    ret = ''
    pos = []
    for idx in idxs:
        ret += source[start+idx]
        pos.append(source_idx_list[start+idx])
    return ret, pos

def extract_sig(Id, top, bottom, idx_list):
    try:
        s1, p1 = extractCharacters(Id, top, bottom, idx_list, "KGVmveHrnvvnlvkwl", [12, 15, 16])
        s2, p2 = extractCharacters(Id, top, bottom, idx_list, "LqfssAysFDaSvweifgaLLnGgt", [3,8,9,10,11,12,13,14,17])
        s3, p3 = extractCharacters(Id, top, bottom, idx_list, "iTvlnltPsl", [4,5])
        s4, p4 = extractCharacters(Id, top, bottom, idx_list, "LrrvlvGGEaL", [4,5,6,7,8])
        s5, p5 = extractCharacters(Id, top, bottom, idx_list, "liNaYGPTEtTVcaTi", [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])

        return s1+s2+s3+s4+s5, p1+p2+p3+p4+p5
    except:
        return '', []

def extract_sig_dict(mydict, add_pos_info=True):
    ret_dict = {}
    #ret_dict = mydict.copy()
    for Id in mydict:
        ret_dict[Id] = {}
        ret_dict[Id]['sig'], pos_info = extract_sig(Id, mydict[Id]['top'], mydict[Id]['bottom'], mydict[Id]['idx_list'])
        if add_pos_info:
            ret_dict[Id]['pos'] = pos_info
    return ret_dict

def check_for_gap_sig(my_sig_dict, print_gap_sig = False, log=False):
    count = 0
    for Id in my_sig_dict:
        pos = my_sig_dict[Id]['pos']
        pos_frac_sum = sum([p-float(int(p)) for p in pos])
        if pos_frac_sum > 0:
            if print_gap_sig:
                print(Id, ':', my_sig_dict[Id]['sig'])
                print(Id, ':', pos)
            count += 1
    if log and count:
        print('There are',count,'signatures with gap among',len(my_sig_dict),'extracted signatures')

def get_aa_alias_dicts(lowercase=True):
    aa_alias_dict = {'A': 'Ala', 'V': 'Val', 'L': 'Leu', 'I': 'Ile', 'P': 'Pro', 'F': 'Phe', 'W': 'Trp', 'M': 'Met', 'K': 'Lys', 'R': 'Arg', 'H': 'His', 'G': 'Gly', 'S': 'Ser', 'T': 'Thr', 'C': 'Cys', 'Y': 'Tyr', 'N': 'Asn', 'Q': 'Gln', 'D': 'Asp', 'E': 'Glu'}
    ret_dict = {}
    ret_inv_dict = {}
    for key in aa_alias_dict:
        if lowercase:
            ret_dict[key.lower()] = aa_alias_dict[key].lower()
            ret_inv_dict[aa_alias_dict[key].lower()] = key.lower()
        else:
            ret_dict[key] = aa_alias_dict[key]
            ret_inv_dict[aa_alias_dict[key]] = key
    return ret_dict, ret_inv_dict

def get_sub_3_lower(sub_in, aa_alias_dict_lower):
    sub_in = sub_in.lower()
    if len(sub_in) == 1:
        return aa_alias_dict_lower[sub_in]
#     elif sub_in.endswith('orn'):
#         return 'orn'
#     elif sub_in.endswith('ala'):
#         return 'ala'
    return sub_in

def get_sub_from_id(sig_dict, print_res=False):
    all_subs = [key.split('|')[1] for key in sig_dict]
    sub_count_dict = sorted([[get_sub_3_lower(x[0], aa_alias_dict_lower), x[1]] for x in dict(Counter(all_subs)).items()], key=lambda x:x[1])
    if print_res:
        print('Original subs', sorted(set(all_subs)))
        print('Modified subs', list(set([x[0] for x in sub_count_dict])))
    return sorted(set(all_subs)), list(set([x[0] for x in sub_count_dict]))

def get_sig_sub_dicts(sig_dict, aa_alias_dict_lower, log=False):
    unique_sigs = set([x['sig'] for x in sig_dict.values()])
    if log:
        print('There are',len(unique_sigs),'unique signatures')

    sig_sub_dict_orig = {}
    for key, val in sig_dict.items():
        if val['sig'] in sig_sub_dict_orig:
            sig_sub_dict_orig[val['sig']].append(get_sub_3_lower(key.split('|')[1], aa_alias_dict_lower))
        else:
            sig_sub_dict_orig[val['sig']] = [get_sub_3_lower(key.split('|')[1], aa_alias_dict_lower)]

    sig_sub_dict = {}
    for key, val in sig_sub_dict_orig.items():
        curr_vals = list(set(val))
        if len(curr_vals) > 1:
            sub_freq_dict = dict(Counter(sig_sub_dict_orig[key]))
            best_subs = [k for k,v in sub_freq_dict.items() if v == max(sub_freq_dict.values())]
            if len(best_subs) == 1:
                sig_sub_dict[key] = best_subs[0]
            elif log:
                print('Conflict for signature',key,'with highest frequency('+str(max(sub_freq_dict.values()))+') subs being '+', '.join(best_subs))
        else:
            sig_sub_dict[key] = curr_vals[0]
        sig_sub_dict_orig[key] = dict(Counter(val))
    if log:
       print('There are',len(sig_sub_dict),'valid training datapoints')

    return sig_sub_dict, sig_sub_dict_orig

def get_sig_dict_from_tsv(file_name):
    with open(file_name, 'r') as f:
        content = f.readlines()
        ret = {}
        for line in content:
            line_split = line.strip('\n').split('\t')
            if len(line_split) != 2 or len(line_split[0]) != 34:
                raise 'Wront format in test tsv, check tabs or signature length (should be 34)'
            ret[line_split[1]] = {}
            ret[line_split[1]]['sig'] = line_split[0]

        return ret

def get_sig_dict_from_file(file_name, sig=False, log=False):
    if sig:
        return get_sig_dict_from_tsv(file_name) 
    
    detail_dict = parse_hmmsearch_output_from_file(file_name, ['aa-activating-core.198-334', 'aroundLys517'])
    a_align_dict = get_hmm_alignment(detail_dict, 'aa-activating-core.198-334')
    a_align_dict_no_indel = removetopindels(a_align_dict)
    sig_dict = extract_sig_dict(a_align_dict_no_indel)
    sig_dict_orig = sig_dict.copy()
    check_for_gap_sig(sig_dict, log=log)
    
    return sig_dict

def get_train_sig_sub_dicts(file_name, aa_alias_dict_lower, sig=False, log=False):
    sig_dict = get_sig_dict_from_file(file_name, sig=sig, log=log)

    # given_subs, modified_subs = get_sub_from_id(sig_dict)

    # sig_sub_dict, sig_sub_dict_orig = get_sig_sub_dicts(sig_dict)
    # all_subs = list(set(sig_sub_dict.values()))
    # print('all subs: '+', '.join(all_subs))
    return get_sig_sub_dicts(sig_dict, aa_alias_dict_lower, log=log)

def get_test_sig_dict(file_name, sig=False, log=False):
    my_dict = get_sig_dict_from_file(file_name, sig=sig, log=log)
    for k in my_dict:
        my_dict[k] = my_dict[k]['sig']
    return my_dict

def get_aa_to_ohe_dict(sig_sub_dict):
    all_aa_in_sig_set = set.union(*[set(s) for s in sig_sub_dict.keys()])
    all_aa_in_sig_set.remove('-')
#     print('All aa in sig:', all_aa_in_sig_set)

    aa_to_ohe_dict = {}
    for i, a in enumerate(all_aa_in_sig_set):
    #     print(i, s)
        aa_to_ohe_dict[a] = [0]*len(all_aa_in_sig_set)
        aa_to_ohe_dict[a][i] = 1
    aa_to_ohe_dict['-'] = [1./len(all_aa_in_sig_set)]*len(all_aa_in_sig_set)

    return aa_to_ohe_dict

def get_ohe_from_sig(sig, aa_to_ohe_dict):
    unidentified = [s for s in sig if s not in aa_to_ohe_dict]
    if unidentified:
        print('Unknown amino acids detected in signature', sig, ':', ', '.join(unidentified))
        return []
    return np.asarray([aa_to_ohe_dict[s] for s in sig]).reshape(-1)

def get_train_data_aa_ohe_dict(sig_sub_dict):
    aa_to_ohe_dict = get_aa_to_ohe_dict(sig_sub_dict)
    train_data = np.asarray([get_ohe_from_sig(s, aa_to_ohe_dict) for s in sig_sub_dict.keys() if len(get_ohe_from_sig(s, aa_to_ohe_dict))])
    
    return train_data, aa_to_ohe_dict

def get_test_data(sig_dict, aa_to_ohe_dict):
    return np.asarray([get_ohe_from_sig(s, aa_to_ohe_dict) for s in sig_dict.values() if len(get_ohe_from_sig(s, aa_to_ohe_dict))])

def train_predict(train_data, train_label, test_data, clf_list, k):
    assert train_data.shape[1] == test_data.shape[1]
    
    tot_test_pred_proba = np.zeros((test_data.shape[0], len(set(train_label))))
    if k > len(set(train_label)):
        print('Reducing input k from', k, 'to', len(set(train_label)), 'due to database constraints')
        k = len(set(train_label))

    for clf in clf_list:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(train_data, train_label)
        test_pred_proba = clf.predict_proba(test_data)
        tot_test_pred_proba += test_pred_proba
    
    return tot_test_pred_proba.argsort()[:, -k:]

def get_pred_from_hmmpfam_output(test_hmmpfam_file_name, k, train_hmmpfam_file_name = '/projects/mohimanilab/romel/AdenPredictor/AdenPred001/data/Adomain_Substrate.fa.hmmpfam2', sig=False, outfile=None, log_train=True, log_test=True):
    aa_alias_dict_lower, inv_aa_alias_dict_lower = get_aa_alias_dicts()

    if log_train:
        print('Train Data Analysis')
    train_sig_sub_dict, train_sig_sub_dict_orig = get_train_sig_sub_dicts(train_hmmpfam_file_name, aa_alias_dict_lower, log=log_train)
    # print(train_sig_sub_dict)
    train_data, aa_to_ohe_dict = get_train_data_aa_ohe_dict(train_sig_sub_dict)
    all_subs_str = sorted(list(set(train_sig_sub_dict.values())))
    all_sub_idx_dict = dict(zip(all_subs_str, range(len(all_subs_str))))
    idx_all_sub_dict = dict(zip(range(len(all_subs_str)), all_subs_str))
    train_label = [all_sub_idx_dict[s] for s in train_sig_sub_dict.values()]
    if log_train:
        print()

    if log_test:
        print('Test Data Analysis')
    test_sig_dict = get_test_sig_dict(test_hmmpfam_file_name, sig=sig, log=log_test)
    test_data = get_test_data(test_sig_dict, aa_to_ohe_dict)

    clf_dict={}
    # clf_dict['lr'] = LogisticRegression(random_state=0, max_iter=1000, multi_class='multinomial', solver='newton-cg', class_weight='balanced')
    clf_dict['rand_for'] = RandomForestClassifier(max_depth=5, criterion='entropy', class_weight='balanced')
    clf_dict['dec_tree'] = DecisionTreeClassifier(random_state=0, criterion='entropy', class_weight='balanced')
    clf_dict['xtra_tree'] = ExtraTreesClassifier(n_estimators=200, random_state=0, criterion='gini', class_weight='balanced')

    test_pred = train_predict(train_data, train_label, test_data, clf_dict.values(), k)
    test_pred_final = []

    for i, s in enumerate(test_sig_dict.values()):
        if s in train_sig_sub_dict_orig:
            known_sub = all_sub_idx_dict[train_sig_sub_dict[s]] if s in train_sig_sub_dict else list(train_sig_sub_dict_orig[s].values())[0]
            final_subs = list(np.flip(test_pred[i]))
            if known_sub in final_subs:
                final_subs.remove(known_sub)
                final_subs = [known_sub] + final_subs
            else:
                final_subs = [known_sub] + final_subs[:-1]
            test_pred_final.append([idx_all_sub_dict[idx] for idx in final_subs])
    
    k = min(k, len(set(train_label)))
    test_out_filename = test_hmmpfam_file_name.split('.')[0] if test_hmmpfam_file_name.split('.')[-1].startswith('hmm') else test_hmmpfam_file_name
    test_out_filename += '_k_' + str(k) + datetime.now().strftime("_%Y%m%d_%H%M%S_%f.tsv")
    test_out_filename = test_out_filename if outfile is None else outfile
    print('Writing output in', test_out_filename)
    with open(test_out_filename, 'w', newline='') as tsvfile:
        fieldnames = ['Id', 'Signature'] + ['Prediction ' + str(i) for i in range(1, k+1)]
        writer = csv.DictWriter(tsvfile, fieldnames=fieldnames, delimiter='\t')
        
        writer.writeheader()
        for i, v in enumerate(test_sig_dict.items()):
            writer.writerow(dict(zip(fieldnames, [v[0], v[1]] + test_pred_final[i])))
    
    return

def print_help():
    print('helper.py usage:')
    print('helper.py -h')
    print('helper.py -i <hmmpfam2 output file to parse> -o <output filepath> -k <number of predictions for each A-domain>')
    print('-i is a required argument')
    print('Predictions are sorted in decreasing order of importance')
    print('Default value of k is 1')
    
def print_error_help():
    print('Please see below for correct usage')
    print()
    print_help()
    print()

def main(argv):
    test_hmmpfam_file_name = None
    train_hmmpfam_file_name = '/projects/mohimanilab/tools/AdenPredictor/data/Adomain_Substrate.fa.hmmpfam2'
    k = 1
    s = 0
    outfile = None
    print()
    # print('Arguments ', argv)
    
    try:
        opts, args = getopt.getopt(argv,"hi:k:o:s:")
    except getopt.GetoptError:
        print_error_help()
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print_help()
            sys.exit()
        elif opt == '-i':
            test_hmmpfam_file_name = arg
        elif opt == '-o':
            outfile = arg
        elif opt == '-s':
            s = int(arg)
        elif opt == '-k':
            try:
                k = int(arg)
            except:
                print('k must be an integer')
                print('Resetting k to default value 1')
                
    
    if test_hmmpfam_file_name is None:
        print('Fatal Error: -i argument is necessary')
        print()
        print_error_help()
        sys.exit(2)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            get_pred_from_hmmpfam_output(test_hmmpfam_file_name, k, train_hmmpfam_file_name=train_hmmpfam_file_name, sig=s, outfile=outfile, log_train=False, log_test=True)
    return

if __name__ == "__main__":
    main(sys.argv[1:])

