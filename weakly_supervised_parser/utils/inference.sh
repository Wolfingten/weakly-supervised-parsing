#!/bin/bash

python3 -m nltk.downloader stopwords

export PYTHONPATH=/nethome/jguertler/weakly-supervised-parsing/

python /nethome/jguertler/weakly-supervised-parsing/weakly_supervised_parser/inference.py \
    --use_inside \
    --model_name_or_path roberta-base \
    --inside_max_seq_length 256 \
    --save_path /data/users/jguertler/outputs/ws_parser/ger/inside.txt \
    --scale_axis 1 \
    --predict_batch_size 32

python /nethome/jguertler/weakly-supervised-parsing/weakly_supervised_parser/inference.py \
    --use_inside_self_train \
    --model_name_or_path roberta-base \
    --inside_max_seq_length 256 \
    --save_path /data/users/jguertler/outputs/ws_parser/ger/inside_self_train.txt \
    --scale_axis 1 \
    --predict_batch_size 32

python /nethome/jguertler/weakly-supervised-parsing/weakly_supervised_parser/inference.py \
    --use_outside \
    --model_name_or_path roberta-base \
    --inside_max_seq_length 256 \
    --save_path /data/users/jguertler/outputs/ws_parser/ger/outside.txt \
    --scale_axis 1 \
    --predict_batch_size 32

python /nethome/jguertler/weakly-supervised-parsing/weakly_supervised_parser/inference.py \
    --use_inside_outside_co_train \
    --model_name_or_path roberta-base \
    --inside_max_seq_length 256 \
    --save_path /data/users/jguertler/outputs/ws_parser/ger/inside_outside_co_train.txt \
    --scale_axis 1 \
    --predict_batch_size 32
