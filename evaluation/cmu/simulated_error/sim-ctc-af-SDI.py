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
	    
    llForward = torch.log(alphas[L-1, T-1] + alphas[L-2, T-1])
	
    return -llForward

def ctc_loss_batch(params, batch_label, blank=0):
    """
    CTC loss function.
    params - n x m matrix of n-D probability distributions(softmax output) over m frames.
    batch_label - batch(N*L) of sequence of phone id's for given example.
    Returns objective, alphas and betas.
    """
    seqLen = batch_label.shape[1] # Length of label sequence (# phones)
    B = batch_label.shape[0]  #batch size
    numphones = params.shape[0] # Number of labels
    L = 2*seqLen + 1 # Length of label sequence with blanks
    T = params.shape[1] # Length of utterance (time)

    alphas = torch.zeros((B,L,T)).double()

    # Initialize alphas and forward pass 
    alphas[:,0,0] = params[blank,0]
    index_init_alphas = torch.stack((batch_label[:,0],torch.ones_like(batch_label[:,0])*0),0).tolist()
    alphas[:,1,0] = params[index_init_alphas]
    
    for t in range(1,T):
        start = max(0,L-2*(T-t)) ##earliest start non-zero state for current t, T-t = how many t left to complete the sequence, for each t, it's able to cover at most 2 labels(directly skip from one non-black state to the blank state)
        for s in range(start,L):
            l = int((s-1)/2)
            # blank
            if s%2 == 0:
                if s==0:
                    alphas[:,s,t] = alphas[:,s,t-1] * params[blank,t]
                else:
                    alphas[:,s,t] = (alphas[:,s,t-1] + alphas[:,s-1,t-1]) * params[blank,t]
            
            else:
                index_list = torch.stack((batch_label[:,l],torch.ones_like(batch_label[:,l])*t),0).tolist()
                condition = torch.logical_or(torch.ones(batch_label.shape[0])==s, batch_label[:,l] == batch_label[:, l-1] )
                ## s-2 will be -1 when s == 1, still valid 
                alphas[:,s,t] = torch.where(condition, (alphas[:,s,t-1] + alphas[:,s-1,t-1]) * params[index_list], \
                                           (alphas[:,s,t-1] + alphas[:,s-1,t-1] + alphas[:,s-2,t-1]) * params[index_list])
	    
    llForward = torch.log(alphas[:,L-1, T-1] + alphas[:,L-2, T-1])
	
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
       
##return only likeli, given the postion for arbitrary state, 
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
    P = params.shape[0] # number of non-blank tokens    

    ##extend the tensor to save "arbitrary state"
    alphas = torch.zeros((L,T,P)).double()

    # Initialize alphas 
    if pos == 0:
        alphas[0,0,0] = params[blank,0]
        # can totally skip the pos
        alphas[2,0,0] = params[blank,0]
        alphas[3,0,0] = params[seq[1],0]
        
        alphas[1,0] = params[0:,0]  #an list all tokens
        alphas[1,0,0] = 0  #can't stay at blank, same as the alphas[0,0,0]
    else:
        alphas[0,0,0] = params[blank,0]
        alphas[1,0,0] = params[seq[0],0]

    for t in range(1,T):
        ###different from v3, +1 below for possible skip paths at the final states
        start = max(0,L-2*(T-t+1)) 
        for s in range(start,L):
            l = int((s-1)/2)
            # blank
            if s%2 == 0:
                if s==0:
                    alphas[s,t,0] = alphas[s,t-1,0] * params[blank,t]
                else:
                    sum = check_arbitrary(alphas, s-1, t-1, [0]) # remove the pathes from blank state, because it's a duplicated path as the first term
                    if sum: ## the first blank(for current t) after the arbitrary state,need to collect the probability from the additional dimension
                        if t == 1:
                            removed = alphas[s,t-1,0] - alphas[s-2,t-1,0] ## should be = 0, totally remove the path because it's the same as the skip path
                            alphas[s,t,0] = (removed + sum + alphas[s-2,t-1,0]) * params[blank,t]
                        else:       
                            removed =  alphas[s,t-1,0] - alphas[s-2,t-2,0] * params[blank,t-1]  ## allow for jump, but only once, same as in v2
                            alphas[s,t,0] = (removed + sum + alphas[s-2,t-1,0] + alphas[s-3,t-1,0]) * params[blank,t]  
                    else:
                        alphas[s,t,0] = (alphas[s,t-1,0] + alphas[s-1,t-1,0]) * params[blank,t]
            elif pos != l and pos != l-1:
                if s == 1 or seq[l] == seq[l-1]:   # the first label or same label twice
                    alphas[s,t,0] = (alphas[s,t-1,0] + alphas[s-1,t-1,0]) * params[seq[l],t]
                else:
                    alphas[s,t,0] = (alphas[s,t-1,0] + alphas[s-1,t-1,0] + alphas[s-2,t-1,0]) \
                        * params[seq[l],t]
            elif pos == l-1: #last token is the arbitrary token, need to collect the probability from the additional dimension, and also consider the skip paths
                sum = check_arbitrary(alphas, s-2, t-1, [0,seq[l]])  ##remove the entry of the blank and the  "l"th token in the last dim, because it's already covered in other terms with the same path
                skip_token = alphas[s-4,t-1,0] * params[seq[l],t]
                skip_empty = alphas[s-3,t-1,0] * params[seq[l],t]
                if l-2 < 0 or seq[l-2] == seq[l]: ###dont allow token skip
                    skip_token = 0
                if t == 1: ## dont allow empty skip
                    skip_empty = 0
                else:
                    ##remove duplicate path1 
                    skip_empty = skip_empty -  alphas[s-3,t-2,0]*params[blank, t-1]*params[seq[l],t]  
                    if s-4 >= 0: 
                        ##remove duplicate path 1 and 2
                        skip_empty = skip_empty -  alphas[s-3,t-2,0]*params[blank, t-1]*params[seq[l],t] -  alphas[s-4,t-2,0]*params[blank, t-1]*params[seq[l],t]
                  
                alphas[s,t,0] = (alphas[s,t-1,0] + alphas[s-1,t-1,0] + sum ) * params[seq[l],t] + skip_empty + skip_token
            else: #current pos can be arbitrary tokens, use the boardcast scale product to allow all the paths       
                if s == 1: #the blank pathes from the first term is already removed for t=0 at initial step, so we don't do it again
                    empty_prob = alphas[s-1,t-1,0] * params[:,t]
                    empty_prob[0] = 0

                    alphas[s,t,:] = (alphas[s,t-1,:].view(1,-1) * params[:,t].view(-1,1)).sum(-1) + empty_prob
                else: #enterting wildcard state, for the skip path and empty path, we need to remove the pos of the same label and blank token to avoid duplicated paths. 
                    skip_prob = alphas[s-2,t-1,0] * params[:,t]  
                    skip_prob[seq[l-1]] = 0    
                    skip_prob[0] = 0    

                    empty_prob = alphas[s-1,t-1,0] * params[:,t]
                    empty_prob[0] = 0

                    alphas[s,t,:] = (alphas[s,t-1,:].view(1,-1) * params[:,t].view(-1,1)).sum(-1) + skip_prob + empty_prob
         
    sum = check_arbitrary(alphas, L-2, T-1)    
    if sum: # last label is arbitrary, inlcludes empty as well so we don't need the last term alphas[L-1,T-1,0], but we need the skip path
        #no need explictly the alphas of T-2 for skip now because in this version we extendted the valid states at time T-1
        llForward = torch.log(sum + alphas[L-3, T-1, 0] + alphas[L-4, T-1, 0])
    else:
        llForward = torch.log(alphas[L-1, T-1, 0] + alphas[L-2, T-1, 0])

    return -llForward

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

        #total_tokens += int(labels.shape[0])
        #print("total_tokens:{}".format(total_tokens))
        ##step 2 run the model, return the post_mat and check the correctness of our loss function
        input_values = processor(row["speech"], return_tensors="pt", sampling_rate=16000).input_values
        return_dict = model(input_values, labels = labels)
        log_like_total = return_dict["loss"].squeeze(0)
        logits = return_dict["logits"].squeeze(0) 
        post_mat = logits.softmax(dim=-1).type(torch.float64)
        
        #step 3  run phoneme replacement and compute GOP
        for order, pid in enumerate(labels):
            ll_denom = ctc_loss_denom(post_mat.transpose(0,1), labels, order, blank=0)
            batch_label = labels.repeat(len(pid_set),1)  #P*L
            
            batch_label[:,order] = torch.Tensor(pid_set)
            ll_num = ctc_loss_batch(post_mat.transpose(0,1), batch_label, blank=0)
            gop_score = -ll_num + ll_denom
            for pid_inner, gop in zip(pid_set,gop_score):
                f.write("{0} {1} {2} {3}\n".format(row['id'],p_tokenizer._convert_id_to_token(int(pid)),p_tokenizer._convert_id_to_token(pid_inner),gop))
                


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






    
  
