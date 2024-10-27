#!/bin/sh

#./process.sh

python3 -m nltk.downloader stopwords

export PYTHONPATH=/nethome/jguertler/weakly-supervised-parsing/
export WANDB_API_KEY="$(cat /nethome/jguertler/.secrets/wandb.key)"

export MODEL_PATH=/data/users/jguertler/models/ws_parser/ger/
export TRAIN_SENTENCES_PATH=/data/users/jguertler/datasets/spmrl/GERMAN_SPMRL/gold/ptb/train/train.German.gold.ptb.tobeparsed.raw

python3 /nethome/jguertler/weakly-supervised-parsing/weakly_supervised_parser/train.py \
    --path_to_train_sentences ${TRAIN_SENTENCES_PATH} \
    --model_name_or_path roberta-base \
    --output_dir ${MODEL_PATH} \
    --max_epochs 10 \
    --lr 5e-6 \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --predict_batch_size 32 \
    --num_workers 16 \
    --inside_max_seq_length 256 \
    --outside_max_seq_length 64 \
    --num_labels 2 \
    --num_self_train_iterations 5 \
    --num_co_train_iterations 2 \
    --scale_axis -1 \
    --upper_threshold 0.995 \
    --lower_threshold 0.005 \
    --num_train_rows 100 \
    --num_valid_examples 100 \
    --seed 42 \
    --run_name "first"
