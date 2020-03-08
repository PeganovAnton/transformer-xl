#!/usr/bin/env bash

python /home/y/transformer-xl/hf_training/launch.py \
--model_type=gpt-2 \
--train_data_file=../data_git/test.txt \
--output_dir=../transformers_test_folder \
--tokenizer_path=../gitbpe-16384.bpe \
--bpe_dropout=0.1 \
--model_size=tiny \
--warmup_tokens=500000 \
--do_train \
--do_eval \
--evaluate_during_training \
--per_gpu_train_batch_size=15 \
--logging_steps=50 \
--eval_data_file=../data_git/valid.txt \
--per_gpu_eval_batch_size=25 \
--save_steps=50 \
--overwrite_output_dir \
--overwrite_cache \
--gradient_accumulation_steps=1 \
--num_train_epochs=1.0 \
--eval_all_checkpoints