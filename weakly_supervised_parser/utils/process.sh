#!/bin/sh

export PYTHONPATH=/home/wolfingten/projects/weakly-supervised-parsing/

python /home/wolfingten/projects/weakly-supervised-parsing/weakly_supervised_parser/utils/process_ptb.py --ptb_path /home/wolfingten/projects/data/spmrl/GERMAN_SPMRL/gold/ptb/ --output_path /home/wolfingten/projects/data/spmrl/GERMAN_SPMRL/gold/ptb/
