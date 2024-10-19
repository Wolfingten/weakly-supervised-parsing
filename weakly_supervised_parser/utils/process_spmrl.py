import os
import re
import string
import argparse
from typing import no_type_check

from weakly_supervised_parser.settings import PTB_SENTENCES_ROOT_DIR, PTB_TREES_ROOT_DIR

from weakly_supervised_parser.settings import (
    PTB_TRAIN_SENTENCES_WITH_PUNCTUATION_PATH,
    PTB_VALID_SENTENCES_WITH_PUNCTUATION_PATH,
    PTB_TEST_SENTENCES_WITH_PUNCTUATION_PATH,
)

from weakly_supervised_parser.settings import (
    PTB_TRAIN_SENTENCES_WITHOUT_PUNCTUATION_PATH,
    PTB_VALID_SENTENCES_WITHOUT_PUNCTUATION_PATH,
    PTB_TEST_SENTENCES_WITHOUT_PUNCTUATION_PATH,
)

from weakly_supervised_parser.settings import (
    PTB_TRAIN_GOLD_WITH_PUNCTUATION_PATH,
    PTB_VALID_GOLD_WITH_PUNCTUATION_PATH,
    PTB_TEST_GOLD_WITH_PUNCTUATION_PATH,
)

from weakly_supervised_parser.settings import (
    PTB_TRAIN_GOLD_WITHOUT_PUNCTUATION_PATH,
    PTB_VALID_GOLD_WITHOUT_PUNCTUATION_PATH,
    PTB_TEST_GOLD_WITHOUT_PUNCTUATION_PATH,
)


def save_with_mod(input_paths, output_paths, func):
    for input, output in zip(input_paths, output_paths):
        with open(input) as i:
            text = i.readlines()

        clean_text = []
        for l in text:
            out = func(l)
            if out:
                clean_text.append(out)

        with open(output, "w") as o:
            o.writelines(clean_text)


#        print(f"Modified: {output}\n{clean_text[:800]}\n")


# TODO: remove tiger syntax info from trees
def delete_tiger(text):
    if text.startswith("(VROOT"):
        no_vroot = re.sub(r"^\(VROOT\s*|\)$", "", text)
    else:
        no_vroot = text
    return re.sub(r"##(.*?)##", "", no_vroot)


# TODO: remove punct from sentences and trees
def remove_punctuation(text):
    no_punct = text.translate(str.maketrans("", "", string.punctuation)).strip(
        " \t\r\f\v"
    )
    return re.sub(r"[ \t\r\f\v]+", " ", no_punct)


def remove_punctuation_trees(text):
    no_punct = re.sub(
        r"\(\$(.*?)\)[ \t\r\f\v]|[ \t\r\f\v]\(\$(.*?)\)|\(\$(.*?)\)", "", text
    )
    if not no_punct.startswith("(S"):
        no_punct = "(S " + no_punct.strip() + ")"
    return no_punct


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ptb_path",
        default="/data/users/jguertler/datasets/spmrl/GERMAN_SPMRL/gold/ptb/",
    )
    args = parser.parse_args()
    save_with_mod(
        [
            PTB_TRAIN_SENTENCES_WITH_PUNCTUATION_PATH,
            PTB_VALID_SENTENCES_WITH_PUNCTUATION_PATH,
            PTB_TEST_SENTENCES_WITH_PUNCTUATION_PATH,
        ],
        [
            PTB_TRAIN_SENTENCES_WITHOUT_PUNCTUATION_PATH,
            PTB_VALID_SENTENCES_WITHOUT_PUNCTUATION_PATH,
            PTB_TEST_SENTENCES_WITHOUT_PUNCTUATION_PATH,
        ],
        remove_punctuation,
    )

    save_with_mod(
        [
            os.path.join(args.ptb_path, "train/train.German.gold.ptb"),
            os.path.join(args.ptb_path, "dev/dev.German.gold.ptb"),
            os.path.join(args.ptb_path, "test/test.German.gold.ptb"),
        ],
        [
            PTB_TRAIN_GOLD_WITH_PUNCTUATION_PATH,
            PTB_VALID_GOLD_WITH_PUNCTUATION_PATH,
            PTB_TEST_GOLD_WITH_PUNCTUATION_PATH,
        ],
        delete_tiger,
    )

    save_with_mod(
        [
            PTB_TRAIN_GOLD_WITH_PUNCTUATION_PATH,
            PTB_VALID_GOLD_WITH_PUNCTUATION_PATH,
            PTB_TEST_GOLD_WITH_PUNCTUATION_PATH,
        ],
        [
            PTB_TRAIN_GOLD_WITHOUT_PUNCTUATION_PATH,
            PTB_VALID_GOLD_WITHOUT_PUNCTUATION_PATH,
            PTB_TEST_GOLD_WITHOUT_PUNCTUATION_PATH,
        ],
        remove_punctuation_trees,
    )
