python3 -m nltk.downloader stopwords

export PYTHONPATH=/nethome/jguertler/weakly-supervised-parsing/

python weakly_supervised_parser/tree/compare_trees.py --tree2 /data/users/jguertler/outputs/ws_parser/inference.txt
