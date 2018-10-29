#!/bin/bash
# a bash script to get a new vast.ai instance ready for the project :)
df -h --total
export KAGGLE_USERNAME=rishabhiitbhu
export KAGGLE_KEY=f8c51d0501658d465f6f72e18b1a4aa5
apt-get -y install python-opencv unzip htop nano
pip install -r requirements.txt

kaggle competitions download -c rsna-pneumonia-detection-challenge -f stage_2_train_images.zip
unzip -q -o stage_1_train_images.zip -d ../data/stage_1_train_images
rm *.zip

kaggle competitions download -c rsna-pneumonia-detection-challenge -f stage_2_test_images.zip
unzip -q -o stage_1_test_images.zip -d ../data/stage_1_test_images
rm *.zip


kaggle competitions download -c rsna-pneumonia-detection-challenge -f stage_2_train_labels.csv 
unzip stage_2_train_labels.csv.zip -d ../data/stage_2_train_labels.csv
kaggle competitions download -c rsna-pneumonia-detection-challenge -f stage_2_detailed_class_info.csv
unzip stage_1_detailed_class_info.csv.zip -d ../data/stage_2_detailed_class_info.csv
kaggle competitions download -c rsna-pneumonia-detection-challenge -f stage_2_sample_submission.csv
mv stage_2_sample_submission.csv ../data/


wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip
unzip ngrok-stable-linux-amd64.zip
rm *.zip
mv ngrok ../
chmod +x ngrok
mkdir data/submission
df -h --total

echo "setting up kaggle key and username"
echo 'export KAGGLE_KEY="f8c51d0501658d465f6f72e18b1a4aa5"' >> ~/.bashrc
echo 'export KAGGLE_USERNAME="rishabhiitbhu"' >> ~/.bashrc



# CUDA_VISIBLE_DEVICES=None tensorboard --logdir=logs/ 
# ./ngrok http 6006

# if locale error in tensorboard, run this: 
# export LC_ALL="C"

# `echo 'export LC_ALL="C"' >> ~/.bashrc`

#kaggle competitions submit rsna-pneumonia-detection-challenge -f submission-0.23.csv -m "23oct model"

# sftp://root@52.204.230.7:16709

# jupyter notebook  --ip=127.0.0.1 --port 8080 --allow-root

# for i in *.zip; do unzip "$i" -d "${i%%.zip}"; done  # one command to extract all zip files in the folder to their respective name folders