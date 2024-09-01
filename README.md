# CTC-based-GOP
This repo related to the paper "A Framework for Phoneme-Level Pronunciation Assessment Using CTC" for INTERSPEECH2024

{xinwei.cao, zijian.fan, torbjorn.svendsen, giampiero.salvi}@ntnu.no

## in the folder:generate-GOP there are the scripts for generating the scalar GOPs

an example call:

python gop-ctc-align-SD.py ./transcriptions/cmu.ctm ./out-models/rCTC-large-from-CTC/checkpoint-6500/  data-new/data-for-w2v2/cmu-kids/metadata.csv ./out-models/rCTC-large-from-CTC/processor_config_gop/ NONE ./test-gop-normnew/test

## in the folder:generate-GOP-features, the script for genearting vector GOPs can be found
an example call:

gop-af-feats.py ./transcriptions/cmu.ctm /home/xinweic/tools-ntnu/wav2vec2/training/out-models/enCTC-large/checkpoint-1250 data-new/data-for-w2v2/cmu-kids/metadata.csv /home/xinweic/tools-ntnu/wav2vec2/training/out-models/enCTC-large/processor_config_gop/ ./out-gops-cmu/cmu-enctc

## in the folder:evaluation, there are the scripts for evaluating the generated GOP/GOP-features on both dataset reported in the paper. 

an example call for evaluating the real errors of CMU-kids:

python analyze_real_auc.py ./gops/cmu_gv3.gop_new uttid.temp label_mapped.txt ./cmu_gv3.json

an example call for evaluating the simulated errors of CMU-kids:

python sim-ctc-af-SDI.py ./cmu.all.ctm ./models/ctc-en/ ./data/local/cmu/train/ ./output/out_cmu_all_gv2.txt

and then

python cal_auc.py --filter ./error_uttid.txt output/out_cmu_all_gv2.txt ./filtered_out_cmu_all_gv2.json

example calls for evaluting the SPO762

python evaluate_gop_scalar.py ./gops/so762_gv1.gop ./metadata.csv ./utt2dur_train ./utt2dur_test
python evaluate_gop_feats.py ./gops/all_gop.feats ./metadata.csv ./utt2dur_train ./utt2dur_test


