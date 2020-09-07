cd transformer-xl-parallel \
  && mkdir -p ../downloads ../data \
  && wget --continue https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip \
        -O ../downloads/wikitext-103-v1.zip \
  && unzip -q -d ../data/ ../downloads/wikitext-103-v1.zip \
  && transformer_xl_path=$(pwd) \
  && cd ../data/wikitext-103 \
  && mv wiki.train.tokens train.txt \
  && mv wiki.valid.tokens valid.txt \
  && mv wiki.test.tokens test.txt \
  && cd ${transformer_xl_path} \
  && pip install -r requirements.txt \
  && echo 'Run training...' \
  && python -m torch.distributed.launch --nproc_per_node 8 train.py \
        --data ../data/wikitext-103 \
        --dataset wt103 \
        --n_layer 12 \
        --d_model 512 \
        --n_head 8 \
        --d_head 64 \
        --d_inner 2048 \
        --dropout 0.1 \
        --dropatt 0.0 \
        --optim adam \
        --lr 0.00025 \
        --tgt_len 512 \
        --mem_len 512 \
        --eval_tgt_len 128 \
        --batch_size 520 \
        --num_gpu 16 \
        --gpu0_bsz 8 \
        --fp16 \
        ${@:2} \

