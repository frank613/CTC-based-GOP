import sys
import re
import os
from sklearn import metrics
import numpy as np
import json
import pandas as pd
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union
import datasets
from transformers.models.wav2vec2 import Wav2Vec2CTCTokenizer, Wav2Vec2Processor
from external.new_modules import Wav2Vec2ForPhoneCE
import torch
from pathlib import Path
import pdb


ds_data_path = '/home/xinweic/cached-data/wav2vec2/data'
ds_cache_path = "/home/xinweic/cached-data/wav2vec2/ds-cache"
datasets.config.DOWNLOADED_DATASETS_PATH = Path(ds_data_path)
datasets.config.HF_DATASETS_CACHE= Path(ds_cache_path)

re_phone = re.compile(r'([@:a-zA-Z]+)([0-9])?(_\w)?')
spec_tokens = set(("<pad>", "<s>", "</s>", "<unk>", "|"))
sil_tokens = set(["sil","SIL","SPN"])

#RE for Teflon files
re_uttid = re.compile(r'(.*/)(.*)\.(.*$)')

#RE for CMU-kids
re_uttid_raw = re.compile(r'(.*)\.(.*$)')

##segmentation, new phonemes are annotated as "_N"
def seg_to_token_seq(p_tokenizer,p_seq):
    segmented = [] #list of pair (pid, start_idx, end_idx) end_idx overlaps the start_idx of the next phoneme
    temp,idx_start = '', 0
    for i,p in enumerate(p_seq):
        if p.endswith('_#'):
            pid = p_tokenizer._convert_token_to_id(p.strip("_#"))
            segmented.append([temp, idx_start, i])
            temp = pid
            idx_start = i
    segmented.append([temp, idx_start, i+1])
    segmented = segmented[1:]
    return segmented


def writes(gop_list, key_list, outFile):
    assert(len(gop_list) == len(key_list))
    #outFile = "./output-gop-nodecode/all_cmu.gop"
    os.makedirs(os.path.dirname(outFile), exist_ok=True)
    with open(outFile, 'w') as fw:
        for key, gop in zip(key_list, gop_list):
            fw.write(key+'\n')
            for cnt, (p,score) in enumerate(gop):
                fw.write("%d %s %.3f\n"%(cnt, p, score))
            fw.write("\n")

def read_align(align_path):
    utt_list =[]
    with open(align_path, "r") as ifile:
        ##to mark different phonemes: e.g "T" in " went to"
        for line in ifile:
            line = line.strip()
            uttid, phonemes = line.split(' ')[0], line.split(' ')[1:]
            uttid = uttid.strip("lbi-")
            p_list = []
            prev_tag = ''
            prev_p = ''
            prev_tone = '' 
            for p in phonemes:
                pure_phoneme = re_phone.match(p).group(1)
                tone = re_phone.match(p).group(2)
                tag = re_phone.match(p).group(3)
                if prev_p != pure_phoneme:
                    p_list.append(pure_phoneme + "_#") 
                elif prev_tone != tone:
                    p_list.append(pure_phoneme + "_#") 
                elif prev_tag != tag:
                    p_list.append(pure_phoneme + "_#") 
                #elif tag == '_I' and prev_tone != tone:
                #    p_list.append(pure_phoneme + "_#") 
                else:
                    p_list.append(pure_phoneme) 
                prev_p = pure_phoneme
                prev_tone = tone
                prev_tag = tag
                #p_list = list(map(lambda x: re_phone.match(x).group(1), phonemes))
            utt_list.append((uttid, p_list))
    return pd.DataFrame(utt_list, columns=('uttid','phonemes')) 

###simplifed, because of frame-wise CE training, no "-" symbols will be outputed
def collect_stats_avg_posterior(post_mat, pid_seq, pid_set, gops_map):
    for order, (pid,start_idx, end_idx) in enumerate(pid_seq):
        length_seg = end_idx - start_idx
        if pid in pid_set:
            for pid_inner in pid_set:
                post_avg = post_mat[start_idx:end_idx][:,pid_inner].mean()
                gops_map[pid][pid_inner].append(post_avg)   


def load_dataset_local_from_dict(csv_path, cache_additional):
    cache_full_path = os.path.join(ds_cache_path, cache_additional)
    if not os.path.exists(cache_full_path):
        datadict = {"audio": []}  
        with open(csv_path) as csvfile:
            next(csvfile)
            for row in csvfile:
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
                batch["p_text"].append(ali_df.loc[ali_df.uttid == uid, "phonemes"].to_list()[0])
        return batch

    ds_map = ds.map(map_to_array, remove_columns=["audio"], batched=True, batch_size=100)
    ds_filtered = ds_map.filter(lambda batch: [ item is not None for item in batch['p_text']], batched=True, batch_size=100, num_proc=3)


    return ds_filtered

    
if __name__ == "__main__":

    print(sys.argv)
    if len(sys.argv) != 6:
        sys.exit("this script takes 5 arguments <cano-alignment file> <w2v2-model-dir> <local-data-csv-folder> <w2v2-preprocessr-dir> <out-file>.\n \
        , it generates the GOP using average posterior of the phoenme recognizer layer of fine-tuned w2v2 model, the csv path must be a folder containing audios files and the csv") 
    #step 0, read the files
    ali_df = read_align(sys.argv[1]) 
    uttid_list = ali_df['uttid'].unique()
    # load the pretrained model and data
    model_path = sys.argv[2]
    csv_path = sys.argv[3]
    prep_path = sys.argv[4]
 
    processor = Wav2Vec2Processor.from_pretrained(prep_path)
    p_tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(prep_path)
    model = Wav2Vec2ForPhoneCE.from_pretrained(model_path)
    model.eval()

    # load dataset and read soundfiles
    ds= load_dataset_local_from_dict(csv_path, "cmu-kids")
    #cuda = torch.device('cuda:1')
    
    #p_set = set(p_tokenizer.encoder.keys()) - spec_tokens
    #count = 0
    with torch.no_grad():
        #pid_set = p_tokenizer.convert_tokens_to_ids(p_set)
        gops_list = []  # gops(|uttid|x|phoneme-seq|)
        key_list= []
        for row in ds:
            #count += 1
            #if count > 10:
            #    break
            if row['id'] not in uttid_list:
                print("ignore uttid: " + row['id'] + ", no alignment can be found")
                continue
            print("processing {0}".format(row['id']))
            #step 1, segmentation (pid_seq = list of (pid, start_idx, end_idx)
            p_seq = ali_df.loc[ali_df.uttid == row["id"], "phonemes"].to_list()[0]
            pid_seq = seg_to_token_seq(p_tokenizer, p_seq)
            #step 2 get the posterior matrix:
            input_values = processor(row["speech"], return_tensors="pt", sampling_rate=16000).input_values
            logits = model(input_values)["logits"].squeeze(0)
            post_mat = logits.softmax(dim=-1)
            #step 3 compute and GOP
            key_list.append(row['id']) 
            gops_list.append([])
            for order, (pid,start_idx, end_idx) in enumerate(pid_seq):
                length_seg = end_idx - start_idx
                post_avg = post_mat[start_idx:end_idx][:,pid].mean()
                gops_list[len(key_list)-1].append((p_tokenizer._convert_id_to_token(pid), torch.log(post_avg)))   
       

    print("done with GOP computation")
    writes(gops_list, key_list, sys.argv[5])






    
  
