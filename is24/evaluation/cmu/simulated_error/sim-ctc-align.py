import sys
import re
import os
from sklearn import metrics
import numpy as np
import json
import pandas as pd
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union
import datasets
from transformers.models.wav2vec2 import Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer, Wav2Vec2Processor
import torch
from pathlib import Path
import pdb



datasets.config.DOWNLOADED_DATASETS_PATH = Path('/localhome/stipendiater/xinweic/wav2vec2/data/downloads')
datasets.config.HF_DATASETS_CACHE= Path('/localhome/stipendiater/xinweic/wav2vec2/data/ds-cache')

re_phone = re.compile(r'([A-Z]+)([0-9])?(_\w)?')
sil_tokens = set(["SIL","SPN"])
spec_tokens = set(("<pad>", "<s>", "</s>", "<unk>", "|"))

xstr = lambda s: s or ""

#RE for CMU files
re_uttid = re.compile(r'(.*/)(.*)\.(.*$)')

##essential for map fucntion to run with multiprocessing, otherwise deadlock, why?
torch.set_num_threads(1)

def auc_cal(array): #input is a nX2 array, with the columns "score", "label"
    labels = [ 0 if i == 0 else 1  for i in array[:, 1]]
    if len(set(labels)) <= 1:
        return "NoDef"
    else:
        #negative because GOP is negatively correlated to the probablity of making an error
        return round(metrics.roc_auc_score(labels, -array[:, 0]),3) 

def read_trans(trans_path):
    trans_map = {}
    cur_uttid = ""
    with open(trans_path, "r") as ifile:
        for line in ifile:
            items = line.strip().split()
            if len(items) != 5:
                sys.exit("input trasncription file must be in the Kaldi CTM format")
            if items[0] != cur_uttid and items[0] not in trans_map: 
                cur_uttid = items[0]
                trans_map[cur_uttid] = []
            phoneme = re_phone.match(items[4]).group(1)
            if phoneme  not in (sil_tokens |spec_tokens):
                trans_map[cur_uttid].append(phoneme)
    return trans_map 

def load_dataset_local_from_dict(folder_path):
    datadict = {"audio": []}  
    with open(folder_path + '/metadata.csv') as csvfile:
        next(csvfile)
        for row in csvfile:
            datadict["audio"].append(folder_path + '/' + row.split(',')[0])
    ds = datasets.Dataset.from_dict(datadict) 
    ds = ds.cast_column("audio", datasets.Audio(sampling_rate=16000))
    #get the array for single row
    def map_to_array(batch):   
        batch["speech"] = [ item["array"] for item in batch["audio"] ]
        batch["p_text"] = []
        batch["id"] = [re_uttid.match(item["path"])[2] for item in batch["audio"]]
        for uid in batch["id"]:
            if uid not in uttid_list:
                batch["p_text"].append(None)
            else:
                batch["p_text"].append(tran_map[uid])
        return batch

    ds_map = ds.map(map_to_array, remove_columns=["audio"], batched=True, batch_size=100)
    #ds_filtered = ds_map
    ds_filtered = ds_map.filter(lambda example: example['p_text'] is not None)
    return ds_filtered

##return the segement of the arbitrary state from the best path(with the largest conrtibution to the denominator), compare it to the forward algrotihm, we don't need to "remove" anything because we take the maximum for viterbi
def viterbi_ctc(params, seq, blank=0):
    """
    CTC loss function.
    params - n x m matrix of n-D probability distributions(softmax output) over m frames.
    seq - sequence of phone id's for given example.
    Returns objective, alphas and betas.
    """
    seqLen = seq.shape[0] # Length of label sequence (# phones)
    numphones = params.shape[0] # Number of labels
    L = 2*seqLen + 1 # Length of label sequence with blanks
    T = params.shape[1] # Length of utterance (time)
    P = params.shape[0] # number of non-blank tokens    

    ##the alphas[s,t] stors the best posterior for the current s at t
    alphas = torch.zeros((L,T)).double()
    
    ##For backtrace, the pointer[s,t] stores the source state of the alphas[s,t]. For t = 0 store -1. 
    #At T+1, the last time step store the winner of final state 0 (only one state (0) valid at T+1)
    pointers = torch.zeros((L,T+1)).double()
    
    # Initialize alphas for viterbi
 
    alphas[0,0] = params[blank,0]
    alphas[1,0] = params[seq[0],0]
    pointers[0,0] = -1
    pointers[1,0] = -1

    for t in range(1,T):
        start = max(0,L-2*(T-t)) 
        for s in range(start,L):
            l = int((s-1)/2)
            # blank
            if s%2 == 0:
                if s==0:
                    alphas[s,t] = alphas[s,t-1] * params[blank,t]
                    pointers[s,t] = s #stays at s=0
                else:         
                    winner = max(alphas[s,t-1], alphas[s-1,t-1])
                    alphas[s,t] = winner * params[blank,t]
                    pointers[s,t] = s if alphas[s,t] == alphas[s,t-1]* params[blank,t] else s-1
            else:
                if l == 0 or seq[l] == seq[l-1]:
                    s0 = alphas[s,t-1]
                    s1 = alphas[s-1,t-1]
                    s2 = -1
                    
                else:
                    s0 = alphas[s,t-1]
                    s1 = alphas[s-1,t-1]
                    s2 = alphas[s-2,t-1]
                winner = max(s0,s1,s2)
                alphas[s,t] = winner * params[seq[l],t]
                if winner == s0: ## stays at s=0
                    pointers[s,t] = s
                elif winner == s1: ## leaving the arbitrary state at t, keep 
                    pointers[s,t] = s-1
                else:
                    pointers[s,t] = s-2

    empty_p = alphas[L-1, T-1]
    final_p = alphas[L-2, T-1]
    winner = max(final_p, empty_p)
    if winner == final_p: 
        pointers[0,T] = L-2
    else:
        pointers[0,T] = L-1             
    return pointers


# return the backtrace path for the current pointer table
def get_backtrace_path(pointers):
    
    T = pointers.shape[1]
    full_path_int = []
    sub_seq = [] ## label's id for the current token
    next_state = 0 #only one state defined for the additional time step
    for t in list(range(T-1,-1,-1)):
        next_state = int(pointers[int(next_state),t])
        full_path_int.append(next_state)
                 
    return (full_path_int)

def single_process(example, pid_set, uttid_list, p_tokenizer, processor, model, out_path):
    row = example   
    #if row['id'] != 'fadf1ao2':
        #return
    #pdb.set_trace()
    pid = str(os.getpid())
    if row['id'] not in uttid_list:
        print("ignore uttid: " + row['id'] + ", no alignment can be found")
        return 0
    print("processing {0}".format(row['id']))
    with torch.no_grad(), open(out_path+"_"+pid+".txt", "a") as f:
        #step 1, get the labels (pid_seq)
        labels = torch.Tensor(p_tokenizer.convert_tokens_to_ids(tran_map[row["id"]]))
        labels = labels.type(torch.int32)
        pid_seq = labels.tolist()
      
        ##step 2 run the model, return the post_mat and check the correctness of our loss function
        input_values = processor(row["speech"], return_tensors="pt", sampling_rate=16000).input_values
        return_dict = model(input_values, labels = labels)
        log_like_total = return_dict["loss"].squeeze(0)
        logits = return_dict["logits"].squeeze(0) 
        post_mat = logits.softmax(dim=-1).type(torch.float64).transpose(0,1)
        pointers = viterbi_ctc(post_mat, labels, blank=0)
        full_path_int = get_backtrace_path(pointers)[:-1]
        full_path_int.reverse()
        
        #step 3  run phoneme replacement and compute GOP
        last_state = 0
        post_count = 0
        post_total = 0  #a tensor of |P| X N
        for i,state in enumerate(full_path_int):
            l = int((last_state - 1)/2) 
            l_new = int((state - 1)/2) 
            if state != last_state:
                if post_count != 0: ##previous state is not blank, token->blank or token1->token2
                    for pid_inner in pid_set:
                        gop = post_total[pid_inner]/post_count
                        f.write("{0} {1} {2} {3}\n".format(row['id'],p_tokenizer._convert_id_to_token(int(pid_seq[l])),p_tokenizer._convert_id_to_token(pid_inner),gop))
                    post_count = 0
                    post_total = 0
                #else: # blank->token
            if state%2 != 0:
                post_count += 1
                post_total += post_mat[:,i] # keep the blank here because we will use pid to index the scores
            last_state = state
        if post_count != 0:
            l = int((last_state - 1)/2)
            for pid_inner in pid_set:
                        gop = post_total[pid_inner]/post_count
                        f.write("{0} {1} {2} {3}\n".format(row['id'],p_tokenizer._convert_id_to_token(int(pid_seq[l])),p_tokenizer._convert_id_to_token(pid_inner),gop))
                   


if __name__ == "__main__":

    print(sys.argv)
    if len(sys.argv) != 5:
        sys.exit("this script takes 4 arguments <transcription file, kaldi-CTM format> <w2v2-model-dir> <local-data-csv-folder> <out-file>.\n \
        , it analyzes the AUC using replacement error, the csv path must be a folder containing audios files and the csv") 
    #step 0, read the files
    tran_map = read_trans(sys.argv[1])
    uttid_list = tran_map.keys()
    # load prior and the pretrained model
    model_path = sys.argv[2]
    csv_path = sys.argv[3]
    #model_path = ""
    processor = Wav2Vec2Processor.from_pretrained("fixed-data-en/processor-en-ctc")
    p_tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("fixed-data-en/processor-en-ctc")
    model = Wav2Vec2ForCTC.from_pretrained(model_path)
    model.eval()
    
    p_set = set(p_tokenizer.get_vocab().keys())
    p_set = p_set - sil_tokens - spec_tokens
    pid_set = p_tokenizer.convert_tokens_to_ids(p_set)

    # load dataset and read soundfiles
    ds = load_dataset_local_from_dict(csv_path)
    ds.map(single_process, fn_kwargs={"pid_set": pid_set, "uttid_list":uttid_list, "p_tokenizer":p_tokenizer, "processor":processor, "model":model, "out_path":sys.argv[4]}, num_proc=10) 
    
                                                   
    print("done with GOP computation")






    
  
