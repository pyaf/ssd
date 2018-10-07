#!/bin/bash
# a bash script to get a new vast.ai instance ready for the project :)
df -h --total
export KAGGLE_USERNAME=rishabhiitbhu
export KAGGLE_KEY=f8c51d0501658d465f6f72e18b1a4aa5
pip install kaggle pydicom tensorboard-logger pandas sklearn tensorboard
apt-get install python-opencv unzip htop nano

kaggle competitions download -c rsna-pneumonia-detection-challenge -f stage_1_train_images.zip
unzip -q -o stage_1_train_images.zip -d stage_1_train_images
rm *.zip

kaggle competitions download -c rsna-pneumonia-detection-challenge -f stage_1_test_images.zip
unzip -q -o stage_1_test_images.zip -d stage_1_test_images
rm *.zip


kaggle competitions download -c rsna-pneumonia-detection-challenge -f stage_1_train_labels.csv.zip
kaggle competitions download -c rsna-pneumonia-detection-challenge -f stage_1_sample_submission.csv.zip
unzip stage_1_train_labels.csv.zip
unzip stage_1_sample_submission.csv.zip

wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip
unzip ngrok-stable-linux-amd64.zip
rm *.zip
mv ngrok ../
chmod +x ../ngrok

# CUDA_VISIBLE_DEVICES=None tensorboard --logdir=logs/ 
# ./ngrok http 6006