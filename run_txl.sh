#/bin/bash

NCCL_MIN_NRINGS=16 NCCL_MAX_NRINGS=16 PYTHONPATH="." python hf_training/launch.py \
	--model_type=txl \
	--example_symbol= \
	--train_data_file=../cooked_data/train_1.txt \
	--output_dir=../cached_dir \
	--tokenizer_path=../gitbpe-16384.bpe \
	--bpe_dropout=0.1 \
	--model_size=txl-like \
	--warmup_tokens=50000000 \
	--weight_decay=0.1 \
	--do_train \
	--do_eval \
	--evaluate_during_training \
	--per_gpu_train_batch_size=5 \
	--logging_steps=500 \
	--eval_data_file=../cooked_data/valid.txt \
	--per_gpu_eval_batch_size=8 \
	--save_steps=500 \
	--num_train_epochs=5.0 \
	--fp16 \
	--fp16_opt_level O1 \
	--overwrite_output_dir \
	--should_continue
