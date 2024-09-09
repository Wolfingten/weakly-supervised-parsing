python3 -m nltk.downloader stopwords

export PYTHONPATH=/nethome/jguertler/weakly-supervised-parsing/

python weakly_supervised_parser/inference.py \
    --use_inside \
    --model_name_or_path roberta-base \
    --inside_max_seq_length 256 \
    --save_path /data/users/jguertler/outputs/ws_parser/inference.txt
