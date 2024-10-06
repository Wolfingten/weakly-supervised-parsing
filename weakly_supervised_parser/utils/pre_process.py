import os
import re
import string

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


## TODO: split data
# def split_data(file_path, ext, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1):
#    with open(file_path, "r", encoding="utf-8") as f:
#        lines = f.readlines()
#
#    total_len = len(lines)
#    train_end = int(train_ratio * total_len)
#    valid_end = train_end + int(valid_ratio * total_len)
#
#    train_set = lines[:train_end]
#    valid_set = lines[train_end:valid_end]
#    test_set = lines[valid_end:]
#
#    directory, _ = os.path.split(file_path)
#
#    with open(
#        os.path.join(directory, f"train_{ext}.txt"), "w", encoding="utf-8"
#    ) as f_train:
#        f_train.writelines(train_set)
#
#    with open(
#        os.path.join(directory, f"valid_{ext}.txt"), "w", encoding="utf-8"
#    ) as f_valid:
#        f_valid.writelines(valid_set)
#
#    with open(
#        os.path.join(directory, f"test_{ext}.txt"), "w", encoding="utf-8"
#    ) as f_test:
#        f_test.writelines(test_set)
#
#    print(
#        f"Data split for {ext}: {len(train_set)} train, {len(valid_set)} valid, {len(test_set)} test"
#    )


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
        print(f"Modified: {output}\n{clean_text[:800]}\n")


# TODO: remove punct from sentences and trees
def remove_punctuation(text):
    no_punct = text.translate(str.maketrans("", "", string.punctuation)).strip(
        " \t\r\f\v"
    )
    return re.sub(r"[ \t\r\f\v]+", " ", no_punct)


def remove_punctuation_trees(text):
    return re.sub(
        r"\(\$(.*?)\)[ \t\r\f\v]|[ \t\r\f\v]\(\$(.*?)\)|\(\$(.*?)\)", "", text
    )


# TODO: remove tiger syntax info from trees
def delete_tiger(text):
    if text.startswith("(VROOT"):
        no_vroot = re.sub(r"^\(VROOT\s*|\)$", "", text)
    else:
        no_vroot = text
    return re.sub(r"##(.*?)##", "", no_vroot)


# TODO: Yoon Kim format

if __name__ == "__main__":
    #    split_data(
    #        os.path.join(PTB_SENTENCES_ROOT_DIR, "train.German.gold.ptb.tobeparsed.raw"),
    #        ext="sentences",
    #    )
    #    split_data(os.path.join(PTB_TREES_ROOT_DIR, "train.German.gold.ptb"), ext="trees")

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
            PTB_TRAIN_GOLD_WITH_PUNCTUATION_PATH,
            PTB_VALID_GOLD_WITH_PUNCTUATION_PATH,
            PTB_TEST_GOLD_WITH_PUNCTUATION_PATH,
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
