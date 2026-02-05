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
PAD_SIL_TOKEN = "SIL"

#RE for SO762
re_uttid = re.compile(r'(.*)\.(.*$)')

##essential for map fucntion to run with multiprocessing, otherwise deadlock, why?
torch.set_num_threads(1)
 
def load_dataset_local_from_dict(folder_path, cache_additional):
    cache_full_path = os.path.join(ds_cache_path, cache_additional)
    if not os.path.exists(cache_full_path):
        datadict = {"audio": [], "p_text":[]}
        with open(folder_path + '/metadata.csv') as csvfile:
            next(csvfile)
            for row in csvfile:
                filename,trans,scores = row.split(',')
                datadict["audio"].append(folder_path + '/' + filename)
                datadict["p_text"].append(trans.split(' '))
        ds = datasets.Dataset.from_dict(datadict) 
        ds = ds.cast_column("audio", datasets.Audio(sampling_rate=16000))
        ds.save_to_disk(cache_full_path)
    ds = datasets.Dataset.load_from_disk(cache_full_path)
    #get the array for single row
    def map_to_array(batch):   
        batch["speech"] = [ item["array"] for item in batch["audio"] ]
        batch["id"] = [re_uttid.match(item["path"])[1] for item in batch["audio"]]
        return batch
    ds_map = ds.map(map_to_array, remove_columns=["audio"], batched=True, batch_size=100, num_proc=5)
    ds_filtered = ds_map.filter(lambda batch: [ item is not None for item in batch['p_text']], batched=True, batch_size=100, num_proc=5)
    
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
    #occ_vec = alphas.sum(dim=1) 	
    return -llForward	

def single_process(batch, p_tokenizer, processor, model_path, out_path):
    proc_id = str(os.getpid())
    model = Wav2Vec2ForCTC.from_pretrained(model_path)
    model.eval()
    with torch.no_grad(), open(out_path+"_"+proc_id+".txt", "a") as f:
        for i,uid in enumerate(batch["id"]):
            print("processing {0}".format(uid))
            f.write(uid+'\n')
            #get the total likelihood of the lable
            input_values = processor(batch["speech"][i], return_tensors="pt", sampling_rate=16000).input_values
            #the default loss here in the config file is "ctc_loss_reduction": "sum" 
            labels = torch.Tensor(p_tokenizer.convert_tokens_to_ids(batch["p_text"][i]))
            labels = labels.type(torch.int32)
            ##return the log_like to check the correctness of our function
            return_dict = model(input_values, labels = labels)
            log_like_total = return_dict["loss"].squeeze(0)
            logits = return_dict["logits"].squeeze(0) 
            post_mat = logits.softmax(dim=-1)

            #compute the GOP-AF-features
            pids = labels.tolist()
            num_token = logits.shape[1]
            for i,pid in enumerate(pids):
                ## The LPP is placed at the first dimension 
                gop_feats = [log_like_total]
                for sub_pid in range(num_token):
                    new_labels = labels.clone().detach()
                    if sub_pid == 0:
                        ##remove the token, for deletion error
                        new_labels = torch.cat([new_labels[:i], new_labels[i+1:]])
                    else:
                        ##for all the substituion errors and correct 
                        new_labels[i] = sub_pid
                    ctc = ctc_loss(post_mat.transpose(0,1), new_labels, blank=0)
                    ##the LPRs  
                    gop_feats.append(-log_like_total+ctc)
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
    
    # load dataset and read soundfiles
    ds= load_dataset_local_from_dict(csv_path, "speechocean762")
    ds.map(single_process, fn_kwargs={"p_tokenizer":p_tokenizer, "processor":processor, "model_path":model_path, "out_path":sys.argv[4]}, batched=True, batch_size=50, num_proc=100) 







    
  
