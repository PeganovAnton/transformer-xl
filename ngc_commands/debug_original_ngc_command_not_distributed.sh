python train.py \
        --data ../data/wikitext-103 \
        --dataset wt103 \
        --n_layer 6 \
        --d_model 512 \
        --n_head 4 \
        --d_head 32 \
        --d_inner 1024 \
        --dropout 0.1 \
        --dropatt 0.0 \
        --optim adam \
        --lr 0.00025 \
        --tgt_len 512 \
        --mem_len 512 \
        --eval_tgt_len 128 \
        --batch_size 520 \
        --num_gpu 2 \
        --gpu0_bsz 8 \
        --fp16 \
        ${@:2} \

