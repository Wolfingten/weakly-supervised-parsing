import torch
from argparse import ArgumentParser
from loguru import logger

from weakly_supervised_parser.settings import TRAINED_MODEL_PATH
from weakly_supervised_parser.utils.prepare_dataset import PTBDataset
from weakly_supervised_parser.model.trainer import InsideOutsideStringClassifier
from weakly_supervised_parser.model.self_trainer import SelfTrainingClassifier
from weakly_supervised_parser.model.co_trainer import CoTrainingClassifier
from weakly_supervised_parser.model.prepare_data import prepare_data_for_self_training, prepare_data_for_co_training, prepare_outside_strings


import pandas as pd
from sklearn.model_selection import train_test_split

from weakly_supervised_parser.tree.helpers import get_constituents, get_distituents
from weakly_supervised_parser.settings import PTB_TRAIN_GOLD_WITHOUT_PUNCTUATION_ALIGNED_PATH
    
parser = ArgumentParser(description="Training Pipeline for the Inside Outside String Classifier", add_help=True)

parser.add_argument("--seed", type=int, default=42, help="Training seed")

parser.add_argument(
    "--path_to_train_sentences", type=str, default="/data/users/jguertler/datasets/ptb_flat/ws_parser/sentences/ptb-train-sentences-without-punctuation.txt", help="Path to pretrained model or model identifier from huggingface.co/models"
)

parser.add_argument(
    "--model_name_or_path", type=str, default="roberta-base", help="Path to pretrained model or model identifier from huggingface.co/models"
)

parser.add_argument("--output_dir", type=str, default=TRAINED_MODEL_PATH, help="Path to the inside/outside model")

parser.add_argument("--max_epochs", type=int, default=10, help="Limits training to a max number number of epochs")

parser.add_argument("--lr", type=float, default=5e-6, help="Learning Rate")

parser.add_argument("--devices", type=int, default=min(1, torch.cuda.device_count()), help="Number of devices to be used by the accelerator for the training strategy")

parser.add_argument(
    "--accelerator", type=str, default="auto", help="Supports passing different accelerator types ('cpu', 'gpu', 'tpu', 'ipu', 'auto')"
)

parser.add_argument("--train_batch_size", type=int, default=32, help="Number of training samples in a batch")

parser.add_argument("--eval_batch_size", type=int, default=32, help="Number of validation samples in a batch")

parser.add_argument("--predict_batch_size", type=int, default=150, help="Parameter for predict_proba")

parser.add_argument("--num_workers", default=16, type=int, help="Number of workers used in the data loader")

parser.add_argument(
    "--inside_max_seq_length", default=256, type=int, help="The maximum total input sequence length after tokenization for the inside model"
)

parser.add_argument(
    "--outside_max_seq_length", default=64, type=int, help="The maximum total input sequence length after tokenization for the outside model"
)

parser.add_argument("--num_labels", default=2, type=int, help="Binary classification, hence two classes")

parser.add_argument("--num_self_train_iterations", default=5, type=int, help="Number of self-training iterations")

parser.add_argument("--num_co_train_iterations", default=2, type=int, help="Number of co-training iterations")

parser.add_argument("--scale_axis", default=-1, type=int, help="Which dimension of output layer to calculate softmax on")

parser.add_argument("--upper_threshold", default=0.99, type=float, help="Threshold value to choose constituents")

parser.add_argument("--lower_threshold", default=0.01, type=float, help="Threshold value to choose distituents")

parser.add_argument("--num_train_rows", default=-1, type=int, help="Subset of training rows for the training")

parser.add_argument("--num_valid_examples", default=-1, type=int, help="Subset of validation samples for the evaluation")

parser.add_argument("--enable_progress_bar", default=True, type=bool, help="Whether to enable progress bar for the Trainer or not")

args = parser.parse_args()

inside_model = InsideOutsideStringClassifier(model_name_or_path=args.model_name_or_path, max_seq_length=args.inside_max_seq_length)

ptb = PTBDataset(data_path=args.path_to_train_sentences)
train, validation = ptb.train_validation_split(seed=args.seed)

inside_model.load_model(pre_trained_model_path=args.output_dir + "inside_model.onnx")
logger.info("Preparing data for self-training!")
train_self_trained, valid_self_trained = prepare_data_for_self_training(
    inside_model=inside_model,
    train_initial=train,
    valid_initial=validation,
    threshold=args.upper_threshold,
    num_train_rows=args.num_train_rows,
    num_valid_examples=args.num_valid_examples,
    seed=args.seed,
    scale_axis=args.scale_axis,
    predict_batch_size=args.predict_batch_size
)