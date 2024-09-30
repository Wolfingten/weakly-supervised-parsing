python3 -m nltk.downloader stopwords

export PYTHONPATH=/nethome/jguertler/weakly-supervised-parsing/

python /nethome/jguertler/weakly-supervised-parsing/weakly_supervised_parser/tree/compare_trees.py --tree2 /data/users/jguertler/outputs/ws_parser/inside.txt

python /nethome/jguertler/weakly-supervised-parsing/weakly_supervised_parser/tree/compare_trees.py --tree2 /data/users/jguertler/outputs/ws_parser/inside_self_train.txt

python /nethome/jguertler/weakly-supervised-parsing/weakly_supervised_parser/tree/compare_trees.py --tree2 /data/users/jguertler/outputs/ws_parser/outside.txt

python /nethome/jguertler/weakly-supervised-parsing/weakly_supervised_parser/tree/compare_trees.py --tree2 /data/users/jguertler/outputs/ws_parser/inside_outside_co_train.txt

