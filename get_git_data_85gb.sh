#!/usr/bin/env bash


if [[ ! -f dataset-v2-py3.tar.gz ]]; then
    wget https://5k-dataset.s3.amazonaws.com/dataset-v2-py3.tar.gz
fi

mkdir -p data/git_85gb
python prepare_git_data.py --tar_path dataset-v2-py3.tar.gz --dataset_path data/git_85gb/ --min_len 50 --seed 30
