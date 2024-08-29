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


ds_data_path = '/home/xinweic/cached-data/wav2vec2/data'
ds_cache_path = "/home/xinweic/cached-data/wav2vec2/ds-cache"
datasets.config.DOWNLOADED_DATASETS_PATH = Path(ds_data_path)
datasets.config.HF_DATASETS_CACHE= Path(ds_cache_path)

re_phone = re.compile(r'([@:a-zA-Z]+)([0-9])?(_\w)?')
spec_tokens = set(("<pad>", "<s>", "</s>", "<unk>", "|"))
sil_tokens = set(["sil", "SIL", "SPN"])

#RE for Teflon files
re_uttid = re.compile(r'(.*/)(.*)\.(.*$)')

#RE for CMU-kids
re_uttid_raw = re.compile(r'(.*)\.(.*$)')

##essential for map fucntion to run with multiprocessing, otherwise deadlock, why?
torch.set_num_threads(1)
    
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

##return only likeli
def ctc_loss(params, seq, blank=0):
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

    alphas = torch.zeros((L,T)).double()
    alpha_bar = torch.zeros(T).double()

    # Initialize alphas and forward pass 
    alphas[0,0] = params[blank,0]
    alphas[1,0] = params[seq[0],0]
    alpha_bar[0] = torch.sum(alphas[:,0])
    alphas[:,0] = alphas[:,0] /  alpha_bar[0]

    for t in range(1,T):
        start = max(0,L-2*(T-t)) 
        end = min(2*t+2,L)
        for s in range(start,L):
            l = int((s-1)/2)
            # blank
            if s%2 == 0:
                if s==0:
                    alphas[s,t] = alphas[s,t-1] * params[blank,t]
                else:
                    alphas[s,t] = (alphas[s,t-1] + alphas[s-1,t-1]) * params[blank,t]
            # same label twice
            elif s == 1 or seq[l] == seq[l-1]:
                alphas[s,t] = (alphas[s,t-1] + alphas[s-1,t-1]) * params[seq[l],t]
            else:
                alphas[s,t] = (alphas[s,t-1] + alphas[s-1,t-1] + alphas[s-2,t-1]) \
                    * params[seq[l],t]
        alpha_bar[t] = torch.sum(alphas[:,t])
        alphas[:,t] = alphas[:,t] / alpha_bar[t]
    
    llForward = torch.log(alpha_bar).sum()   
	
    return -llForward

##check if the last dim > 0, return the sum of last dimension (collect the posterior for each possible tokens),the zero_pos is excluded in the sum.
##zero pos here starst from 0def check_arbitrary(in_alphas, s, t, zero_pos=[]):
def check_arbitrary(in_alphas, s, t, zero_pos=[]):
    if torch.count_nonzero(in_alphas[s,t]) > 1:
        if len(zero_pos) != 0:
            mask = torch.ones_like(in_alphas[s,t])
            for i in zero_pos:
                mask[i] = 0
            return sum(in_alphas[s,t][mask.bool()])
        else:
            return sum(in_alphas[s,t][:])
    else:
        return False

##make the deleted labels, returning both the original and deleted lables, the second) 
def make_deletion(labels, pos):
  
    return [labels, torch.cat((labels[:pos], labels[pos+1:]))]

# def get_alpha_bar(alphas, t, blank, next_label_idx, pos):
#     ## for comupting the alpha bar, we need to remove the blank state and next_label state in the arbitrary state  
#     ###exclude the same state in the "ärbitrary" state when computing the alpha_bar
#     arbitrary_state = 2*pos + 1 
#     alpha_mask = torch.ones(alphas.shape[2], dtype=torch.bool)
#     alpha_mask[blank] = False
#     if next_label_idx is not None:
#         alpha_mask[next_label_idx] = False
#     return alphas[:arbitrary_state,t,0].sum() + alphas[arbitrary_state+1:,t,0].sum() + alphas[arbitrary_state,t,alpha_mask].sum()


def get_alpha_bar(alphas, t, blank, pos):
    arbitrary_state = 2*pos + 1 
    alpha_mask = torch.ones(alphas.shape[2], dtype=torch.bool)
    alpha_mask[blank] = False
    return alphas[:arbitrary_state,t,0].sum() + alphas[arbitrary_state+1:,t,0].sum() + alphas[arbitrary_state,t,alpha_mask].sum()

##This version is free of deletion, so no need to consider the skip paths, easy for the normalized alphas
def ctc_loss_denom(params, seq, pos, blank=0):
    """
    CTC loss function.
    params - n x m matrix of n-D probability distributions(softmax output) over m frames.
    seq - sequence of phone id's for given example.
    Returns objective, alphas and betas.
    """
    seqLen = seq.shape[0] # Length of label sequence (# phones)
    L = 2*seqLen + 1 # Length of label sequence with blanks
    T = params.shape[1] # Length of utterance (time)
    P = params.shape[0] # number of tokens    

    ## constraint mask for disabling insertion, and in this version we don't allow phoneme->blank but remains in the arbitrary state 
    mask_ins = torch.eye(P)
    #mask_ins[blank,:] = torch.ones(P)
    
    ##extend the tensor to save "arbitrary state"
    alphas = torch.zeros((L,T,P)).double()
    alpha_bar = torch.zeros(T)
    # if pos == len(seq) - 1:
    #     next_label_idx = None
    # else:
    #     next_label_idx = seq[pos+1]

    
    # Initialize alphas 
    if pos == 0:
        alphas[0,0,0] = params[blank,0]     
        alphas[1,0] = params[:,0]  #an list all tokens
        alphas[1,0,0] = 0  #can't stay at blank, same as the alphas[0,0,0] 
    else:
        alphas[0,0,0] = params[blank,0]
        alphas[1,0,0] = params[seq[0],0]
    
    alpha_bar[0] = get_alpha_bar(alphas, 0, blank, pos)
    alphas[:,0,:] = alphas[:,0,:] /  alpha_bar[0]
    
    for t in range(1,T):
        start = max(0,L-2*(T-t)) 
        for s in range(start,L):
            l = int((s-1)/2)
            # blank
            if s%2 == 0:
                if s==0:
                    alphas[s,t,0] = alphas[s,t-1,0] * params[blank,t]
                else:
                    sum = check_arbitrary(alphas, s-1, t-1, [blank]) # remove the pathes from blank state, because it's a duplicated path as the first term
                    if sum: ## the first blank(for current t) after the arbitrary state,need to collect the probability from the additional dimension
                        alphas[s,t,0] = (alphas[s,t-1,0] + sum) * params[blank,t]
                    else:
                        alphas[s,t,0] = (alphas[s,t-1,0] + alphas[s-1,t-1,0]) * params[blank,t]
            elif pos != l and pos != l-1:
                if s == 1 or seq[l] == seq[l-1]:   # the first label or same label twice
                    alphas[s,t,0] = (alphas[s,t-1,0] + alphas[s-1,t-1,0]) * params[seq[l],t]
                else:
                    alphas[s,t,0] = (alphas[s,t-1,0] + alphas[s-1,t-1,0] + alphas[s-2,t-1,0]) \
                        * params[seq[l],t]
            elif pos == l-1: #last token is the arbitrary token, need to collect the probability from the additional dimension, and also consider the skip paths
                sum = check_arbitrary(alphas, s-2, t-1, [blank,seq[l]])  ##remove the entry of the blank and the  "l"th token in the last dim, there will be no blank prob in this version, and seq[l] must go to blank first 
                alphas[s,t,0] = (alphas[s,t-1,0] + alphas[s-1,t-1,0] + sum ) * params[seq[l],t]
            else: #current pos can be arbitrary tokens, use the boardcast scale product to allow all the paths    
                if s == 1: #the blank pathes from the first term is already removed for t=0 at initial step, so we don't do it again
                    empty_prob = alphas[s-1,t-1,0] * params[:,t]
                    empty_prob[blank] = 0

                    alphas[s,t,:] = (alphas[s,t-1,:].view(1,-1) * params[:,t].view(-1,1) * mask_ins).sum(-1) + empty_prob
                    
                else: #enterting wildcard state, for the skip path and empty path, we need to remove the pos of the same label and blank token to avoid duplicated paths. alph
                    skip_prob = alphas[s-2,t-1,0] * params[:,t]  
                    skip_prob[seq[l-1]] = 0    
                    skip_prob[blank] = 0    

                    empty_prob = alphas[s-1,t-1,0] * params[:,t]
                    empty_prob[blank] = 0

                    alphas[s,t,:] = (alphas[s,t-1,:].view(1,-1) * params[:,t].view(-1,1) * mask_ins).sum(-1) + skip_prob + empty_prob
                    
       
        ### for blank state 
        #pdb.set_trace()
        alpha_bar[t] = get_alpha_bar(alphas, t, blank, pos)
        alphas[:,t,:] = alphas[:,t,:] / alpha_bar[t]
    
    llForward = torch.log(alpha_bar).sum() 
    return -llForward

def single_process(example, p_tokenizer, processor, model, out_path):
    row = example
    proc_id = str(os.getpid())
    print("processing {0}".format(row['id']))
    with torch.no_grad(), open(out_path+"_"+proc_id+".txt", "a") as f:
        f.write(row['id']+'\n')
        #get the total likelihood of the lable
        input_values = processor(row["speech"], return_tensors="pt", sampling_rate=16000).input_values
        #the default loss here in the config file is "ctc_loss_reduction": "sum" 
        labels = torch.Tensor(p_tokenizer.convert_tokens_to_ids(tran_map[row["id"]]))
        labels = labels.type(torch.int32)
        ##return the log_like to check the correctness of our function
        return_dict = model(input_values, labels = labels)
        logits = return_dict["logits"].squeeze(0) 
        post_mat = logits.softmax(dim=-1).type(torch.float64)
        ll_self = ctc_loss(post_mat.transpose(0,1), labels, blank=0)
        #step 2, compute the GOP
        pids = labels.tolist()
        for i,pid in enumerate(pids):
            label_list = make_deletion(labels, i)
            assert len(label_list) == 2
            ll_denom = ctc_loss_denom(post_mat.transpose(0,1), label_list[0], i, blank=0)
            gop = -ll_self + ll_denom
            f.write("%d %s %s\n"%(i, p_tokenizer._convert_id_to_token(int(pid)), gop.item()))
        f.write("\n")

        

if __name__ == "__main__":

    print(sys.argv)
    if len(sys.argv) != 6:
        sys.exit("this script takes 5 arguments <transcription file, kaldi-CTM format> <w2v2-model-dir> <local-data-csv-folder> <w2v2-preprocessor-dir> <out-file>.\n \
        , it generates the GOP using a fine-tuned w2v2 CTC model, the csv path must be a folder containing audios files and the csv") 
    #step 0, read the files
    tran_map = read_trans(sys.argv[1]) 
    uttid_list = tran_map.keys()
    # load the pretrained model and data
    model_path = sys.argv[2]
    csv_path = sys.argv[3]
    prep_path = sys.argv[4]
 
    processor = Wav2Vec2Processor.from_pretrained(prep_path)
    p_tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(prep_path)
    model = Wav2Vec2ForCTC.from_pretrained(model_path)
    model.eval()

    # load dataset and read soundfiles
    ds= load_dataset_local_from_dict(csv_path, "cmu-kids")
    ds.map(single_process, fn_kwargs={"p_tokenizer":p_tokenizer, "processor":processor, "model":model, "out_path":sys.argv[5]}, num_proc=1) 
    
    print("done")
    
    
       







    
  
