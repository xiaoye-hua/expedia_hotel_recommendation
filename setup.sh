#!/bin/bash
# Step
alias python='python3'
alias pip='pip3'

# step 1: install; refer to https://cloud.google.com/compute/docs/gpus/install-drivers-gpu#installation_scripts
curl https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.py --output install_gpu_driver.py
sudo python3 install_gpu_driver.py


# step 2: validate with DL code
sudo apt-get install python3-pip # python3-dev  libcupti-dev
sudo pip3 install tensorflow-gpu  #  tf 1.4 ????
# validate code https://github.com/bharat3012/Google-Cloud-An-Easy-Way-to-an-Amazing-Platform/edit/master/gpu/gpu_mnist_speedcheck.py



# step: code
git clone https://github.com/xiaoye-hua/expedia_hotel_recommendation.git
cd expedia_hotel_recommendation/

python3 -m pip install --upgrade pip
python3 -m pip install --upgrade Pillow
# Step setup kaggle APi
sudo apt install unzip
pip3 install kaggle

export KAGGLE_USERNAME=huaguo
export KAGGLE_KEY=e7690011b6e9f0d34d6fce6ef615f37e

~/.local/bin/kaggle competitions download -c expedia-personalized-sort
unzip expedia-personalized-sort.zip
unzip data.zip
mkdir raw_data
mv *.csv raw_data
rm *.zip

# Step: run code
cat requirements.txt | xargs -n 1 pip3 install   # pip install skip error: https://stackoverflow.com/a/28795395/9734266
export PYTHONPATH=./:PYTHONPATH


# run code
mkdir data/raw_data/big_data
mkdir data/raw_data/small_data
python scripts/reduce_file_size.py
python3 scripts/create_offline_features.py
python3 scripts/data_cvt.py
python3 scripts/model_train.py




# basic setup: git, conda  -> can't found conda
sudo apt-get update
sudo apt-get install wget
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
rm Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc


conda install scikit-learn pandas jupyter ipython
conda create -n expedia_project python=3.9
conda activate expedia_project
export PYTHONPATH=./:PYTHONPATH
