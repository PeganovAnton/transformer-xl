#!/usr/bin/env bash


wget https://5k-dataset.s3.amazonaws.com/11kk.tar.gz
mkdir data/git
tar -xf 11kk.tar.gz -C data/git

python prepare_git_data.py --pattern 'data/git/full-dataset/*.py' --dataset_path data/git/ --min_len 50
