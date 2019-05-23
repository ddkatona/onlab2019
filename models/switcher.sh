#!bin/bash
wget https://storage.googleapis.com/disentanglement_lib/unsupervised_study_v1/$1.zip
unzip $1.zip
rm $1.zip
mv $1/ current/
