# Segmentation-Free-GOP
This repo relates to the paper "A Framework for Phoneme-Level Pronunciation Assessment Using CTC"   
https://www.isca-archive.org/interspeech_2024/cao24b_interspeech.pdf

and its extension "Segmentation-Free Goodness of Pronunciation"     
https://ieeexplore.ieee.org/document/11355844

Author List: {xinwei.cao, zijian.fan, torbjorn.svendsen, giampiero.salvi}@ntnu.no

- The older files related to our first paper are moved to the folder [is24] for the purpose of reproducing the results in the first paper
- The folder [taslpro26] corresponds to our second paper. The folder contains detailed implementations of GOP-SF-CTC-Norm and its feature vector version. Note that for the feature vectors, the additional normalizing term is extraced from the output of GOP-SF-CTC-Norm method.
- We also provide a pseudocode in the same folder "GOP_CTC_SF_pseudo.pdf" to illustrate how to implment one of the methods (GOP-CTC-SD-Norm). It should be helpful for understanding other methods as well. 

