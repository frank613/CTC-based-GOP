import sys
import pdb
import numpy as np
import pandas as pd
from utils import balanced_sampling
from concurrent.futures import ProcessPoolExecutor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import re
import matplotlib.pyplot as plt
from scipy import stats

re_phone = re.compile(r'([@:a-zA-Z]+)([0-9])?(_\w)?') # can be different for different models
opt_SIL = 'SIL' ##can be different for different models
poly_order = 2

def round_score(score, floor=0.1, min_val=0, max_val=2):
    score = np.maximum(np.minimum(max_val, score), min_val)
    return np.round(score / floor) * floor

def readGOP(gop_file, p_table):
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
                #assert( len(label_phoneme) == len(seq_score))
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
        cur_score = float(fields[2])
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

def train_model_for_phone(gops, labels, p):
    #figure = plt.figure()
    #axe = plt.gca()
    #axe.scatter(gops,labels)
    #label_ref = np.unique(labels).tolist()
    #data = [ gops[labels == l] for l in label_ref] 
    #axe.boxplot(data,  0, 'rs', 0, labels = label_ref)
    #plt.savefig("./out-plot-gv1/" + f"{p}")
    model = LinearRegression()
    labels = labels.reshape(-1, 1)
    gops = gops.reshape(-1, 1)
    gops = PolynomialFeatures(poly_order).fit_transform(gops)
    gops, labels = balanced_sampling(gops, labels)
    model.fit(gops, labels)
    return model



if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit("this script takes 4 arguments <GOP file> <metadata.csv> <train-utt2dur-kaldiformat> <test-utt2dur-kaldiformat>. It labels the phonemes in the GOP file based on the annotation file, learns a polynomial regression model, predict the test set, outputs a summary ")

    #readfiles
    p_dict = read_anno(sys.argv[2])
    data_list = readGOP(sys.argv[1], p_dict)
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
    df = pd.DataFrame(gop_table, columns=('phoneme','score','label','uttid', "isTrain"))
    p_set = df['phoneme'].unique()
    if len(p_set) != 39:
        sys.exit("phoneme number is not 39, check the files")

    ##training
    train_data_of = {}
    for p in p_set:
        records = df.loc[(df["phoneme"] == p) & (df["isTrain"] == True), ["score","label"]] 
        n_array = records.to_numpy()
        scores, labels = n_array[:,0], n_array[:,1].astype(int)
        train_data_of.setdefault(p,(scores,labels))

    model_of = {}
    for p, (scores,labels) in train_data_of.items():
        r_model = train_model_for_phone(scores,labels, p)
        model_of.setdefault(p, r_model)
    # Train polynomial regression
    #with ProcessPoolExecutor(10) as ex:
    #    futures = [(p,ex.submit(train_model_for_phone, scores,labels, p)) for p, (scores,labels) in train_data_of.items()] 
    #    model_of = {p: future.result() for p, future in futures} 

    # Evaluate
    test_data_of = {}
    for p in p_set:
        records_eva = df.loc[(df["phoneme"] == p) & (df["isTrain"] == False), ["score","label"]]
        n_array_eva = records_eva.to_numpy()
        scores_eva, labels_eva = n_array_eva[:,0], n_array_eva[:,1].astype(int)
        test_data_of.setdefault(p,(scores_eva,labels_eva))


    all_results = np.empty((0,2))
    for p,(gops_eva, ref) in test_data_of.items():
        gops_eva = PolynomialFeatures(poly_order).fit_transform(gops_eva.reshape(-1,1))
        model = model_of[p]
        hyp = model.predict(gops_eva).reshape(-1)
        #hyp = np.round(hyp)
        hyp = round_score(hyp,1)
        #print(np.unique(hyp, return_counts=True))
        results = np.stack((ref, hyp), axis = 1)
        all_results = np.concatenate((all_results,results))


    # summary
    print(f'MSE: {metrics.mean_squared_error(all_results[:,0], all_results[:,1]):.2f}')
    print(f'Corr: {np.corrcoef(all_results[:,0], all_results[:,1])[0][1]:.2f}')
    res = stats.pearsonr(all_results[:,0], all_results[:,1])
    res_tuple = res.confidence_interval(confidence_level=0.95)
    low, high = res_tuple.low, res_tuple.high
    print(f'Corr-v2: {res.statistic:.2f}, low={low:.2f}, high={high:.2f}')
    print(metrics.classification_report(all_results[:,0].astype(int), all_results[:,1].astype(int)))
    print(confusion_matrix(all_results[:,0].astype(int), all_results[:,1].astype(int)))



        

    
