import sys
import re
import os
from sklearn import metrics
import numpy as np
import json
import pandas as pd
import pdb
import argparse

def auc_cal(array): #input is a nX2 array, with the columns "score", "label"
    labels = [ 0 if i == 0 else 1  for i in array[:, 1]]
    if len(set(labels)) <= 1:
        return "NoDef"
    else:
        #negative because GOP is negatively correlated to the probablity of making an error
        return round(metrics.roc_auc_score(labels, -array[:, 0]),3)

def read_rep(file_path, error_list=None):
    if error_list is None:
        error_list = []
    list_tuple = []
    with open(file_path, "r") as ifile:
        for line in ifile:
            fields = line.strip().split(' ')
            if len(fields) != 4:
                sys.exit("wrong input line")
            if fields[0] in error_list:
                continue
            list_tuple.append(fields[1:3] + [round(float(fields[3]),3)])

        return pd.DataFrame(list_tuple, columns=['origin','replaced', 'gop'])



def write(df, outFile):
    out_form = { \
                'phonemes':{},
                'summary': {"average-mean-diff": None, "average-AUC": None, "total_real": None, "total_error":None}}
    #count of phonemes}

    total_real = 0
    total_error = 0
    total_auc = 0
    total_mean_diff = 0
    total_entry = 0

    p_set = df["origin"].unique() 
    for p_out in p_set:
        real_arr = df.loc[(df["origin"] == p_out) & (df["replaced"] == p_out), "gop"].to_numpy(dtype=np.float)
        if len(real_arr) == 0:
            continue
        real_label = np.stack((real_arr, np.full(len(real_arr), 0)), 1)
        scores = []
        total_real += len(real_label)
        for p_in in p_set:
            if p_in == p_out:
                continue
            sub_arr = df.loc[(df["origin"] == p_in) & (df["replaced"] == p_out),"gop"].to_numpy(dtype=np.float)
            if len(sub_arr) == 0:
                continue
            sub_label = np.stack((sub_arr, np.full(len(sub_arr), 1)), 1)
            auc_value = auc_cal(np.concatenate((real_label, sub_label)))
            if auc_value != "NoDef":
                auc_value = round(auc_value, 3)
            scores.append((p_in, sub_arr.mean(), len(sub_arr), auc_value))
            total_error += len(sub_arr)

        if len(scores) == 0:
            continue
        total_entry += 1
        confused_pid, p_mean, num_error, auc = sorted(scores, key = lambda x: x[3])[0]
        mean_diff = round(real_arr.mean() - p_mean, 3)
        out_form["phonemes"][p_out] = (confused_pid, mean_diff, auc, len(real_arr), num_error)
        total_auc += auc
        total_mean_diff += mean_diff
    out_form["summary"]["average-mean-diff"]=total_mean_diff/total_entry
    out_form["summary"]["average-AUC"]=total_auc/total_entry
    out_form["summary"]["total_real"]=total_real
    out_form["summary"]["total_error"]=total_error

    os.makedirs(os.path.dirname(outFile), exist_ok=True)
    with open(outFile, "w") as f:
        json.dump(str(out_form), f)


def read_filter(in_file):
    if in_file is None:
        return None
    with open(in_file, 'r') as inF:
        error_list = []
        for line in inF:
            fields = line.strip().split()
            if len(fields) != 1:
                    print("wrong line:{}".format(fields))
                    sys.exit("wrong input line")
            error_list.append(fields[0])
    return error_list


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='it analyzes the AUC given the txt file ')
    parser.add_argument('sub_file')
    parser.add_argument('out_json_file')
    parser.add_argument('--filter', help="filter file with uttid list that are to be excluded from the computation")

    args = parser.parse_args()
    #step 0, read the files
    filter_list = read_filter(args.filter)
    df = read_rep(args.sub_file, filter_list)
    write(df, args.out_json_file)
