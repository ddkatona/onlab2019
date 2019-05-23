#!bin/bash
cd models
wget https://storage.googleapis.com/disentanglement_lib/unsupervised_study_v1/$1.zip
unzip $1.zip
rm $1.zip
rm -r current
mv $1 current
cd ..
rm -r results/evaluated$1/
dlib_evaluate --input_dir=postprocessed300k --output_dir=results/evaluated$1 --gin_config=disentanglement_lib/config/unsupervised_study_v1/metric_configs/hungarian.gin
eog mx_test.png
