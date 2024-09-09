# CTC-based-GOP
This repo relates to the paper "A Framework for Phoneme-Level Pronunciation Assessment Using CTC" for INTERSPEECH2024 <br />
{xinwei.cao, zijian.fan, torbjorn.svendsen, giampiero.salvi}@ntnu.no

## In the folder:[generate-GOP] there are the scripts for generating the scalar GOPs
An example call: <br />
python gop-ctc-align-SD.py ./transcriptions/cmu.ctm ./out-models/rCTC-large-from-CTC/checkpoint-6500/  data-new/data-for-w2v2/cmu-kids/metadata.csv ./out-models/rCTC-large-from-CTC/processor_config_gop/ NONE ./test-gop-normnew/test

## In the folder [generate-GOP-features], the script for genearting vector GOPs can be found
An example call: <br />
gop-af-feats.py ./transcriptions/cmu.ctm /home/xinweic/tools-ntnu/wav2vec2/training/out-models/enCTC-large/checkpoint-1250 data-new/data-for-w2v2/cmu-kids/metadata.csv /home/xinweic/tools-ntnu/wav2vec2/training/out-models/enCTC-large/processor_config_gop/ ./out-gops-cmu/cmu-enctc

## In the folder [evaluation], there are the scripts for evaluating the generated GOP/GOP-features on both dataset reported in the paper. 
An example call for evaluating the real errors of CMU-kids:<br />
python analyze_real_auc.py ./gops/cmu_gv3.gop_new uttid.temp label_mapped.txt ./cmu_gv3.json <br />

An example call for evaluating the simulated errors of CMU-kids: <br />
python sim-ctc-af-SDI.py ./cmu.all.ctm ./models/ctc-en/ ./data/local/cmu/train/ ./output/out_cmu_all_gv2.txt

And then: <br />
python cal_auc.py --filter ./error_uttid.txt output/out_cmu_all_gv2.txt ./filtered_out_cmu_all_gv2.json

Example calls for evaluting the Speakocean762: <br />
python evaluate_gop_scalar.py ./gops/so762_gv1.gop ./metadata.csv ./utt2dur_train ./utt2dur_test <br />
python evaluate_gop_feats.py ./gops/all_gop.feats ./metadata.csv ./utt2dur_train ./utt2dur_test

## The essential data related files for running the script can be found in the folder [data] and [metadata]

## The ASR-CTC training script is provided in the folder [ctc-ASR-training]
