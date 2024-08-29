import sys
import pdb
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from collections import Counter
import re
import random

re_phone = re.compile(r'([@:a-zA-Z]+)([0-9])?(_\w)?') # can be different for different models
opt_SIL = 'SIL' ##can be different for different models
poly_order = 1

def add_more_negative_data(data):
    # Put all examples together
    whole_data = []
    for ph in data:
        for feats,label in zip(*data[ph]):
            whole_data.append((ph,feats,label))

    # Take the 2-score examples of other phones as the negative examples
    for cur_ph in data:
        feats, labels = data[cur_ph]
        count_of_label = Counter(labels)
        example_number_needed = 2 * count_of_label[2] - len(labels)
        if example_number_needed > 0:
            features = random.sample([feats for ph, feats, label in whole_data
                                      if ph != cur_ph and label == 2],
                                     example_number_needed)
            data[cur_ph][0] = data[cur_ph][0] + features
            data[cur_ph][1] = data[cur_ph][1] + [0] * example_number_needed
    return data

def round_score(score, floor=0.1, min_val=0, max_val=2):
    score = np.maximum(np.minimum(max_val, score), min_val)
    return np.round(score / floor) * floor

def readGOP_feats(gop_file, p_table):
    in_file = open(gop_file, 'r')
    isNewUtt = True
    skip = False
    seq_score = []
    df_temp = []
    label_phoneme = []
    for line in in_file:
        line = line.strip()
        fields = line.split(' ')
        if isNewUtt:
            if len(fields) != 1:
                sys.exit("uttid must be the first line of each utterance")
            uttid=fields[0]
            skip = False
            if uttid not in p_table:
                skip = True
            else:
                label_phoneme = p_table[uttid][1]
            isNewUtt = False
            continue
        if line == '':
            if not skip:
                ## length in the gop file must the same as len(anno)
                assert( len(label_phoneme) == len(seq_score))
                if len(label_phoneme) != len(seq_score):
                    pdb.set_trace()
                    sys.exit()
                df_temp.append((uttid, [(p,g,l) for (p,g),l in zip(seq_score, label_phoneme)]))
            seq_score = []
            label_phoneme = []
            isNewUtt = True
            continue
        if len(fields) != 3:
            continue
        cur_match = re_phone.match(fields[1])
        if (cur_match):
            cur_phoneme = cur_match.group(1)
        else:
            sys.exit("non legal phoneme found in the gop file")
        cur_score = [float(x) for x in fields[2].split(',')]
        ####optional silence
        if cur_phoneme != opt_SIL:
            seq_score.append((cur_phoneme, cur_score))
    return df_temp

def read_anno(anno_file):
    p_dict = {}
    with open(anno_file, 'r') as inf:
        next(inf)
        for row in inf:
            filename,trans,scores = row.strip().split(',')
            filename = filename.split('.')[0]
            trans = trans.split(' ')
            scores = [ float(i) for i in scores.split(' ')]
            p_dict.setdefault(filename, (trans, scores))
    return p_dict

def readList(file_path):
    uttlist = []
    with open(file_path, 'r') as inf:
        for line in inf:
            fields = line.strip().split()
            if len(fields) != 2:
                sys.exit("illegal line found in the anno file")
            uttlist.append(fields[0])
    return uttlist

def train_model_for_phone(feats, labels):
    model = SVR()
    labels = labels.ravel()
    model.fit(feats, labels)
    return model

if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit("this script takes 4 arguments <GOP-feats file> <metadata.csv> <train-utt2dur-kaldiformat> <test-utt2dur-kaldiformat>. It labels the phonemes in the GOP file based on the annotation file, learns a SVR model, predict the test set, outputs a summary ")

    #readfiles
    p_dict = read_anno(sys.argv[2])
    data_list = readGOP_feats(sys.argv[1], p_dict)
    train_list = readList(sys.argv[3])
    test_list = readList(sys.argv[4])
    
    gop_table=[]
    for item in data_list:
        if item[0] in train_list:
            isTrain = True
        elif item[0] in test_list:
            isTrain = False
        else:
            sys.exit("found uttid not in train nor test")
        for itm in item[1]:
            gop_table.append(list(itm) + [item[0], isTrain])
    df = pd.DataFrame(gop_table, columns=('phoneme','feats','label','uttid', "isTrain"))
    p_set = df['phoneme'].unique()
    if len(p_set) != 39:
        sys.exit("phoneme number is not 39, check the files")

    ##training
    train_data_of = {}
    for p in p_set:
        records = df.loc[(df["phoneme"] == p) & (df["isTrain"] == True), ["feats","label"]] 
        feats = records["feats"].tolist()
        labels = records["label"].tolist()
        train_data_of.setdefault(p,[feats,labels])

    # Make the dataset more blance
    #pdb.set_trace()
    train_data_of = add_more_negative_data(train_data_of)


    # Train polynomial regression
    with ProcessPoolExecutor(10) as ex:
        futures = [(p,ex.submit(train_model_for_phone, np.array(feats), np.array(labels))) for p, (feats,labels) in train_data_of.items()] 
        model_of = {p: future.result() for p, future in futures} 

    # Evaluate
    test_data_of = {}
    for p in p_set:
        records_eva = df.loc[(df["phoneme"] == p) & (df["isTrain"] == False), ["feats","label"]]
        feats_eva = np.array(records_eva["feats"].tolist())
        labels_eva = np.array(records_eva["label"].tolist())
        test_data_of.setdefault(p,(feats_eva,labels_eva))


    all_results = np.empty((0,2))
    for p,(feats_eva, ref) in test_data_of.items():
        model = model_of[p]
        hyp = model.predict(feats_eva)
        hyp = round_score(hyp,1)
        results = np.stack((ref, hyp), axis = 1)
        all_results = np.concatenate((all_results,results))


    # summary
    print(f'MSE: {metrics.mean_squared_error(all_results[:,0], all_results[:,1]):.2f}')
    print(f'Corr: {np.corrcoef(all_results[:,0], all_results[:,1])[0][1]:.2f}')

    print(metrics.classification_report(all_results[:,0].astype(int), all_results[:,1].astype(int)))
    print(confusion_matrix(all_results[:,0].astype(int), all_results[:,1].astype(int)))

        

    
