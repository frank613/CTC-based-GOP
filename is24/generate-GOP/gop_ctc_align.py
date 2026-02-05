import sys
import re
import os
from sklearn import metrics
import numpy as np
import json
import pandas as pd
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union
import datasets
from transformers.models.wav2vec2 import Wav2Vec2CTCTokenizer, Wav2Vec2Processor,Wav2Vec2ForCTC
import torch
from pathlib import Path
import pdb
import matplotlib.pyplot as plt

ds_data_path = '/home/xinweic/cached-data/wav2vec2/data'
ds_cache_path = "/home/xinweic/cached-data/wav2vec2/ds-cache"
datasets.config.DOWNLOADED_DATASETS_PATH = Path(ds_data_path)
datasets.config.HF_DATASETS_CACHE= Path(ds_cache_path)

re_phone = re.compile(r'([@:a-zA-Z]+)([0-9])?(_\w)?')
#spec_tokens = set(("<pad>", "<s>", "</s>", "<unk>", "|"))
sil_tokens = set(["sil","SIL","SPN"])
spec_tokens = set(("<pad>", "<s>", "</s>", "<unk>", "|"))
PAD_SIL_TOKEN = "SIL"

#RE for Teflon files
re_uttid = re.compile(r'(.*/)(.*)\.(.*$)')

#RE for CMU-kids
re_uttid_raw = re.compile(r'(.*)\.(.*$)')


def writes(gops_list, outFile):
    os.makedirs(os.path.dirname(outFile), exist_ok=True)
    with open(outFile, 'w') as fw:
        for key, gop_list in gops_list:
            fw.write(key+'\n')
            for cnt, (p,score) in enumerate(gop_list):
                fw.write("%d %s %.3f\n"%(cnt, p, score))
            fw.write("\n")
    
#pad_sil_token on begin and end of the sequence if not None   
def read_trans(trans_path, pad_sil_token=None):
    trans_map = {}
    cur_uttid = ""
    with open(trans_path, "r") as ifile:
        for line in ifile:
            items = line.strip().split()
            if len(items) != 5:
                sys.exit("input trasncription file must be in the Kaldi CTM format")
            if items[0] != cur_uttid and items[0] not in trans_map: 
                if pad_sil_token: ##add SIL at the begining and end of the sequence 
                    if cur_uttid != "":
                        trans_map[cur_uttid].append(pad_sil_token)
                    cur_uttid = items[0]
                    trans_map[cur_uttid] = [pad_sil_token]
                else:
                    cur_uttid = items[0]
                    trans_map[cur_uttid] = []
                cur_uttid = items[0]
            phoneme = re_phone.match(items[4]).group(1)                
            if phoneme not in (sil_tokens | spec_tokens):
                trans_map[cur_uttid].append(phoneme)
    return trans_map 

def load_dataset_local_from_dict(csv_path, cache_additional):
    cache_full_path = os.path.join(ds_cache_path, cache_additional)
    if not os.path.exists(cache_full_path):
        datadict = {"audio": []}  
        #with open(folder_path + '/metadata.csv') as csvfile:
        with open(csv_path) as csvfile:
            next(csvfile)
            for row in csvfile:
                #datadict["audio"].append(folder_path + '/' + row.split(',')[0])
                datadict["audio"].append(row.split(',')[0])
        ds = datasets.Dataset.from_dict(datadict) 
        ds = ds.cast_column("audio", datasets.Audio(sampling_rate=16000))
        ds.save_to_disk(cache_full_path)
    ds = datasets.Dataset.load_from_disk(cache_full_path)
    #get the array for single row
    def map_to_array(batch):   
        batch["speech"] = [ item["array"] for item in batch["audio"] ]
        batch["p_text"] = []
        batch["id"] = [re_uttid_raw.match(item["path"])[1] for item in batch["audio"]]
        for uid in batch["id"]:
            if uid not in uttid_list:
                batch["p_text"].append(None)
            else:
                batch["p_text"].append(tran_map[uid])
        return batch
    ds_map = ds.map(map_to_array, remove_columns=["audio"], batched=True, batch_size=100)
    ds_filtered = ds_map.filter(lambda batch: [ item is not None for item in batch['p_text']], batched=True, batch_size=100, num_proc=3)
    #ds_filtered = ds_map.filter(lambda example: example['p_text'] is not None)

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
    


   

if __name__ == "__main__":

    print(sys.argv)
    if len(sys.argv) != 7:
        sys.exit("this script takes 6 arguments <transcription file, kaldi-CTM format> <w2v2-model-dir> <local-data-csv-folder> <w2v2-preprocessr-dir> <SIL-token> <out-file>.\n \
            , it analyzes the GOP using the ctc-align methods, the csv path must be a folder containing audios files and the metadata.csv. SIL indicates the token used for pad the SIL at the BOS/EOS") 
    #step 0, read the files
    tran_path = sys.argv[1]
    model_path = sys.argv[2]
    csv_path = sys.argv[3]
    prep_path = sys.argv[4]
    sil_token = sys.argv[5]
 
    #step 0, read the files
    if sil_token == PAD_SIL_TOKEN:
        tran_map = read_trans(tran_path, pad_sil_token=PAD_SIL_TOKEN) 
    else:
        tran_map = read_trans(tran_path) 
    uttid_list = tran_map.keys()   
    processor = Wav2Vec2Processor.from_pretrained(prep_path)
    p_tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(prep_path)
    model = Wav2Vec2ForCTC.from_pretrained(model_path)
    model.eval()
   
    # load dataset and read soundfiles
    ds= load_dataset_local_from_dict(csv_path, "cmu-kids")
    #cuda = torch.device('cuda:1')
    
    #p_set = set(p_tokenizer.encoder.keys()) - spec_tokens - sil_tokens
    count = 0
    with torch.no_grad():
        #pid_set = p_tokenizer.convert_tokens_to_ids(p_set)
        gops_list = []  # (uttid, (phoneme, scores))
        for row in ds:
            #count += 1
            #if count > 10:
                #break
            #if row['id'] != 'fabm2cy2':
                #continue
            print("processing {0}".format(row['id']))
            #get the total likelihood of the lable
            input_values = processor(row["speech"], return_tensors="pt", sampling_rate=16000).input_values
            #the default loss here in the config file is "ctc_loss_reduction": "sum" 
            labels = torch.Tensor(p_tokenizer.convert_tokens_to_ids(row["p_text"]))
            labels = labels.type(torch.int32)
            ##return the log_like to check the correctness of our function
            return_dict = model(input_values, labels = labels)
            log_like_total = return_dict["loss"].squeeze(0)
            logits = return_dict["logits"].squeeze(0) 
            post_mat = logits.softmax(dim=-1).type(torch.float64).transpose(0,1)
      
            #step 2, compute the GOP
            pointers = viterbi_ctc(post_mat, labels, blank=0)
            full_path_int = get_backtrace_path(pointers)[:-1]
            full_path_int.reverse()
            pids = labels.tolist()
            gop_list = []
            last_state = 0
            post_count = 0
            post_total = 0
            for i,state in enumerate(full_path_int):
                l = int((last_state - 1)/2) 
                l_new = int((state - 1)/2) 
                if state != last_state:
                    if post_count != 0: ##previous state is not blank, token->blank or token1->token2
                        gop_list.append((p_tokenizer._convert_id_to_token(pids[l]), torch.log(post_total/post_count)))
                        post_count = 0
                        post_total = 0
                    #else: # blank->token
                if state%2 != 0:
                    post_count += 1
                    post_total += post_mat[pids[l_new],i]
                last_state = state
            if post_count != 0:
                l = int((last_state - 1)/2)
                gop_list.append((p_tokenizer._convert_id_to_token(pids[l]), torch.log(post_total/post_count)))
            gops_list.append((row['id'], gop_list))
 
       

    print("done with GOP computation")
    writes(gops_list, sys.argv[6])
            
    

   







    
  
