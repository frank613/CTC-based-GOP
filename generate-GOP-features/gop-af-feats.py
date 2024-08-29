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

datasets.config.DOWNLOADED_DATASETS_PATH = Path('/home/xinweic/exps/downloads')
datasets.config.HF_DATASETS_CACHE= Path('/home/xinweic/exps/Cache')

re_phone = re.compile(r'([@:a-zA-Z]+)([0-9])?(_\w)?')
sil_tokens = set(["SIL","SPN"])
spec_tokens = set(("<pad>", "<s>", "</s>", "<unk>", "|"))

#RE for Teflon files
re_uttid = re.compile(r'(.*/)(.*)\.(.*$)')

##essential for map fucntion to run with multiprocessing, otherwise deadlock, why?
torch.set_num_threads(1)
 
def load_dataset_local_from_dict(folder_path):
    datadict = {"audio": [], "p_text":[]}  
    with open(folder_path + '/metadata.csv') as csvfile:
        next(csvfile)
        for row in csvfile:
            filename,trans,scores = row.split(',')
            datadict["audio"].append(folder_path + '/' + filename)
            datadict["p_text"].append(trans.split(' '))
    ds = datasets.Dataset.from_dict(datadict) 
    ds = ds.cast_column("audio", datasets.Audio(sampling_rate=16000))
    #get the array for single row
    def map_to_array(batch):   
        batch["speech"] = [ item["array"] for item in batch["audio"] ]
        batch["id"] = [re_uttid.match(item["path"])[2] for item in batch["audio"]]
        return batch

    ds_map = ds.map(map_to_array, remove_columns=["audio"], batched=True, batch_size=100)
    ds_filtered = ds_map.filter(lambda example: example['p_text'] is not None)
    #ds_filtered = ds_map

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

    # Initialize alphas and forward pass 
    alphas[0,0] = params[blank,0]
    alphas[1,0] = params[seq[0],0]

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
	    
    forward_prob = (alphas[L-1, T-1] + alphas[L-2, T-1])
	
    return forward_prob

##check if the last dim > 0, return the sum of last dimension (collect the posterior for each possible tokens),the zero_pos is excluded in the sum.
##zero pos must "-1" because we already remove the blank token in the last dimension
def check_arbitrary(in_alphas, s, t, zero_pos=None):
    if torch.count_nonzero(in_alphas[s,t]) > 1:
        if zero_pos:
            mask = torch.ones_like(in_alphas[s,t])
            mask[zero_pos] = 0
            return sum(in_alphas[s,t][mask.bool()])
        else:
            return sum(in_alphas[s,t][:])
    else:
        return False
    
##return only likeli, given the postion for arbitrary state
def ctc_loss_denom(params, seq, pos, blank=0):
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
    P = params.shape[0] - 1 # number of non-blank tokens    

    ##extend the tensor to save "arbitrary state"
    alphas = torch.zeros((L,T,P)).double()

    # Initialize alphas and forward pass 
    alphas[0,0,0] = params[blank,0]
    if pos == 0:
        alphas[1,0] = params[1:,0]  #an list of non-blank 
    else:
        alphas[1,0,0] = params[seq[0],0]

    for t in range(1,T):
        start = max(0,L-2*(T-t)) 
        ##end = min(2*t+2,L)
        for s in range(start,L):
            l = int((s-1)/2)
            # blank
            if s%2 == 0:
                if s==0:
                    alphas[s,t,0] = alphas[s,t-1,0] * params[blank,t]
                else:
                    sum = check_arbitrary(alphas, s-1, t-1)
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
            elif pos == l-1: #last token is the arbitrary token, need to collect the probability from the additional dimension
                sum = check_arbitrary(alphas, s-2, t-1, seq[l]-1)  ##remove the entry of the "l"th token in the last dim, because it's not allowed for a direct transfer for dublicated label
                alphas[s,t,0] = (alphas[s,t-1,0] + alphas[s-1,t-1,0] + sum) * params[seq[l],t]
            else: #current pos can be non-blank arbitrary tokens, keep the same token if already in the state of t-1
                skip_prob = alphas[s-2,t-1,0] * params[1:,t]  
                skip_prob[seq[l] - 1] = 0   #need to remove the pos of the same label,because it's not allowed to skip for duplicated labels 
                alphas[s,t,:] = (alphas[s,t-1,:] + alphas[s-1,t-1,0]) * params[1:,t] + skip_prob
         


    sum = check_arbitrary(alphas, L-2, T-1)    
    if sum: # last label is arbitrary
        llForward = torch.log(sum + alphas[L-1, T-1, 0])
    else:
        llForward = torch.log(alphas[L-1, T-1, 0] + alphas[L-2, T-1, 0])
	
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
        labels = torch.Tensor(p_tokenizer.convert_tokens_to_ids(row["p_text"]))
        labels = labels.type(torch.int32)
        ##return the log_like to check the correctness of our function
        return_dict = model(input_values, labels = labels)
        log_like_total = return_dict["loss"].squeeze(0)
        logits = return_dict["logits"].squeeze(0) 
        post_mat = logits.softmax(dim=-1)

        #step 2, compute the GOP
        pids = labels.tolist()
        for i,pid in enumerate(pids):
            gop_feats = [log_like_total]
            new_labels = labels.clone().detach()
            new_labels = torch.cat([new_labels[:i], new_labels[i+1:]])
            ctc = ctc_loss(post_mat.transpose(0,1), new_labels, blank=0)
            gop_feats.append(-torch.log(ctc))
            gop_feats.append(-log_like_total-torch.log(ctc))
            feat_s = ",".join([ str(torch.round(feat,decimals=3).numpy()) for feat in gop_feats])
            f.write("%d %s %s\n"%(i, p_tokenizer._convert_id_to_token(int(pid)), feat_s))
        f.write("\n")
     
if __name__ == "__main__":

    print(sys.argv)
    if len(sys.argv) != 5:
        sys.exit("this script takes 4 arguments <w2v2-model-dir> <local-data-csv-folder> <w2v2-preprocessor-dir> <out-file>.\n \
        , it generates the GOP using a fine-tuned w2v2 CTC model, the csv path must be a folder containing audios files and the csv") 
    #step 0, read the files
    # load the pretrained model and data
    model_path = sys.argv[1]
    csv_path = sys.argv[2]
    prep_path = sys.argv[3]
    
 
    processor = Wav2Vec2Processor.from_pretrained(prep_path)
    p_tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(prep_path)
    model = Wav2Vec2ForCTC.from_pretrained(model_path)
    model.eval()
    
    # p_set = set(p_tokenizer.get_vocab().keys())
    # p_set = p_set - sil_tokens - spec_tokens
    # pid_set = p_tokenizer.convert_tokens_to_ids(p_set)

    # load dataset and read soundfiles
    ds= load_dataset_local_from_dict(csv_path)
    ds.map(single_process, fn_kwargs={"p_tokenizer":p_tokenizer, "processor":processor, "model":model, "out_path":sys.argv[4]}, num_proc=10) 
    #cuda = torch.device('cuda:1')






    
  
