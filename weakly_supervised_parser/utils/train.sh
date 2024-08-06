export PYTHONPATH=/nethome/jguertler/weakly-supervised-parsing/

export MODEL_PATH=/data/users/jguertler/models/ws_parser/
export TRAIN_SENTENCES_PATH=/data/users/jguertler/datasets/ptb_flat/ws_parser/sentences/ptb-train-sentences-without-punctuation.txt

python3 /nethome/jguertler/weakly-supervised-parsing/weakly_supervised_parser/train.py \
    --path_to_train_sentences ${TRAIN_SENTENCES_PATH} \
    --model_name_or_path roberta-base \
    --output_dir ${MODEL_PATH} \
    --max_epochs 10 \
    --lr 5e-6 \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --predict_batch_size 1 \
    --num_workers 16 \
    --inside_max_seq_length 256 \
    --outside_max_seq_length 64 \
    --num_labels 2 \
    --num_self_train_iterations 5 \
    --num_co_train_iterations 2 \
    --scale_axis 0 \
    --upper_threshold 0.995 \
    --lower_threshold 0.005 \
    --num_train_rows 100 \
    --num_valid_examples 100 \
    --seed 42