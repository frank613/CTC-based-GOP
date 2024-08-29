import sys
import re
import pandas as pd
sys.path.append('/home/xinweic/tools-ntnu/edit-distance')
import edit 
import pdb
import numpy as np
from sklearn import metrics
import json
import os


re_phone = re.compile(r'([A-Z]+)[0-9]*(_\w)*|<pad>')
#store all the sub pairs
#for each phone in the GOP file:
    #label "S" if it's a substitution error based on the other input files.
    #label "D" if it's a deletion error based on the other input files.
    #label "C" if it's the same in both files

def auc_cal(array): #input is a nX2 array, with the columns "score", "label"
    labels = [ 0 if i == 'C' else 1  for i in array[:, 1]]
    if len(set(labels)) <= 1:
        return "NoDef"
    else:
        #negative because GOP is negatively correlated to the probablity of making an error
        rvalue = metrics.roc_auc_score(labels, -array[:, 0])
        return round(rvalue,3)


def labelError(GOP_file, error_list, tran_file):
    gop_df = readGOPToDF(GOP_file)
    tran_df = readTRANToDF(tran_file)
    df = pd.DataFrame(columns=('phonemes','scores','labels', 'uttid'))
    tran_list = tran_df['uttid'].unique()
    gop_list = gop_df['uttid'].unique()
    extended = []
    sub_list = []
    #outer loop is the cano df because it is extracted from the gop file and they should have the same number of uttids
    for index, row in gop_df.iterrows(): 
        uttid = row['uttid']
        print("processing {}".format(uttid))
        if uttid not in tran_list:
            print("warning: uttid {0} can't be found in the transcription".format(uttid))
            continue
        cano_seq = [ k for k,v in row['seq-score']]
        tran_df_filtered = tran_df.loc[tran_df["uttid"] == uttid, "seq"]
        if len(tran_df_filtered) != 1:
            sys.exit("duplicate uttids detected in the transcription file, check the input")
        tran_seq = tran_df_filtered.tolist()[0] #[0] converts 2d list to 1d
        dist, labels = edit.edit_dist(cano_seq, tran_seq)
        sub_list += edit.get_sub_pair_list(cano_seq, tran_seq, labels)
        if dist == 0 or uttid not in error_list:
            labels_resized = ['C'] * len(labels)
        else:
            labels_resized = [ label for idx, label in enumerate(labels) if label != 'I']
            if len(labels_resized) != len(cano_seq):
                sys.exit("length of edit distance not maching the gop")

        extended += [ pair + (labels_resized[idx], uttid) for idx, pair in enumerate(row['seq-score']) ]
    df = pd.concat([df, pd.DataFrame(extended, columns=['phonemes','scores','labels', 'uttid'])])
    #json
    #p:(auc_value, frequent_sub, mean, std, count_of_del, count_of_sub, total_count)
    out_form = { \
                'phonemes':{},
                'summary': {"average-mean": None, "average-std": None, "average-AUC": None}}
    
    #pdb.set_trace()
    p_replace_set = np.append(df['phonemes'].unique(), '*')
    pair_dict = {phoneme_out: { phoneme_in: 0 for phoneme_in in p_replace_set} for phoneme_out in p_replace_set}
    for pair in sub_list:
        l,r = pair.split(' -> ')
        if l not in p_replace_set or r not in p_replace_set:
            continue
        pair_dict[l][r] += 1

    total_mean=0
    total_std=0
    total_auc=0
    for phoneme in df['phonemes'].unique():
        data_false = df.loc[(df["phonemes"] == phoneme) & (df["labels"] == 'C'), ['scores', 'labels']].to_numpy()
        data_true = df.loc[(df["phonemes"] == phoneme) & (df["labels"] != 'C'), ['scores','labels']].to_numpy()
        auc_value = auc_cal(np.concatenate((data_true, data_false), axis=0))
        total_auc += auc_value
        sorted_items = sorted(pair_dict[phoneme].items(), key=lambda kv: kv[1], reverse=True)
        freq_sub =  sorted_items[0][0] if sorted_items[0][0]!='*' else sorted_items[1][0]
        mean = df.loc[(df["phonemes"] == phoneme),"scores"].mean()
        total_mean += mean
        std = df.loc[(df["phonemes"] == phoneme),"scores"].std()
        total_std += std
        num_del = len(df.loc[(df["phonemes"] == phoneme) & (df["labels"] == 'D')])
        num_sub = len(df.loc[(df["phonemes"] == phoneme) & (df["labels"] == 'S')])
        num_total = len(df.loc[df["phonemes"] == phoneme])
        out_form["phonemes"][phoneme]=(auc_value, freq_sub, mean, std, num_del, num_sub, num_total)

    num_phonemes = len(df['phonemes'].unique())
    out_form["summary"]["average-mean"]=total_mean/num_phonemes
    out_form["summary"]["average-std"]=total_std/num_phonemes
    out_form["summary"]["average-AUC"]=total_auc/num_phonemes

    return out_form



    


def readGOPToDF(ark_file):
    in_file = open(ark_file, 'r')
    df = pd.DataFrame(columns=('uttid', 'seq-score'))
    isNewUtt = True
    seq_score = []
    for line in in_file:
        line = line.strip()
        fields = line.split(' ')
        if isNewUtt:
            if len(fields) != 1:
                sys.exit("uttid must be the first line of each utterance")
            uttid=fields[0]
            isNewUtt = False
            continue
        if line == '':
            df.loc[len(df.index)] = [uttid, seq_score]
            isNewUtt = True
            seq_score = []
            continue
        if len(fields) != 3:
            continue
        cur_match = re_phone.match(fields[1])
        if (cur_match):
            cur_phoneme = cur_match.group(1)
        else:
            sys.exit("non legal phoneme found in the gop file")
        cur_score = float(fields[2]) 
        seq_score.append((cur_phoneme, cur_score)) 
    return df

def readTRANToDF(tran_file):
    in_file=open(tran_file, 'r')
    df = pd.DataFrame(columns=('uttid', 'seq'))
    for line in in_file:
        uttid,seq = line.split()
        df.loc[len(df.index)] = [uttid, seq.split(';')]
    return df



if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit("this script takes 4 arguments <GOP file> <error-uttid-list>  <transcribed phoneme-seq file> <outJson>. It labels the phonemes in the GOP file and output a summary in json format")

    utt_list = []
    with open(sys.argv[2]) as ifile:
            for line in ifile:
                line = line.strip()
                fields = line.split()
                if len(fields) != 1:
                    sys.exit("wrong input line")
                utt_list.append(fields[0])
    json_dict = labelError(sys.argv[1], utt_list, sys.argv[3])
    os.makedirs(os.path.dirname(sys.argv[4]), exist_ok=True)
    with open(sys.argv[4], "w") as f:
        json.dump(json_dict, f)

