PROJECT_DIR = "/home/wolfingten/projects/weakly_supervised_parser/"

PTB_TREES_ROOT_DIR = "/home/wolfingten/projects/data/spmrl/GERMAN_SPMRL/gold/ptb/"
PTB_SENTENCES_ROOT_DIR = "/home/wolfingten/projects/data/spmrl/GERMAN_SPMRL/gold/ptb/"

PTB_TRAIN_SENTENCES_WITH_PUNCTUATION_PATH = (
    PTB_SENTENCES_ROOT_DIR + "train/train.German.gold.ptb.tobeparsed.raw"
)
PTB_VALID_SENTENCES_WITH_PUNCTUATION_PATH = (
    PTB_SENTENCES_ROOT_DIR + "dev/dev.German.gold.ptb.tobeparsed.raw"
)
PTB_TEST_SENTENCES_WITH_PUNCTUATION_PATH = (
    PTB_SENTENCES_ROOT_DIR + "test/test.German.gold.ptb.tobeparsed.raw"
)

PTB_TRAIN_SENTENCES_WITHOUT_PUNCTUATION_PATH = (
    PTB_SENTENCES_ROOT_DIR + "train/train_sentences_without_punct.txt"
)
PTB_VALID_SENTENCES_WITHOUT_PUNCTUATION_PATH = (
    PTB_SENTENCES_ROOT_DIR + "dev/valid_sentences_without_punct.txt"
)
PTB_TEST_SENTENCES_WITHOUT_PUNCTUATION_PATH = (
    PTB_SENTENCES_ROOT_DIR + "test/test_sentences_without_punct.txt"
)

PTB_TRAIN_GOLD_WITH_PUNCTUATION_PATH = (
    PTB_TREES_ROOT_DIR + "train/train_gold_with_punct.txt"
)
PTB_VALID_GOLD_WITH_PUNCTUATION_PATH = (
    PTB_TREES_ROOT_DIR + "dev/dev_gold_with_punct.txt"
)
PTB_TEST_GOLD_WITH_PUNCTUATION_PATH = (
    PTB_TREES_ROOT_DIR + "test/test_gold_with_punct.txt"
)

PTB_TRAIN_GOLD_WITHOUT_PUNCTUATION_PATH = (
    PTB_TREES_ROOT_DIR + "train/train_trees_without_punct.txt"
)
PTB_VALID_GOLD_WITHOUT_PUNCTUATION_PATH = (
    PTB_TREES_ROOT_DIR + "dev/valid_trees_without_punct.txt"
)
PTB_TEST_GOLD_WITHOUT_PUNCTUATION_PATH = (
    PTB_TREES_ROOT_DIR + "test/test_trees_without_punct.txt"
)

PTB_TRAIN_GOLD_WITHOUT_PUNCTUATION_ALIGNED_PATH = (
    PTB_TREES_ROOT_DIR + "train/train_trees_without_punct.txt"
)
PTB_VALID_GOLD_WITHOUT_PUNCTUATION_ALIGNED_PATH = (
    PTB_TREES_ROOT_DIR + "dev/valid_trees_without_punct.txt"
)
PTB_TEST_GOLD_WITHOUT_PUNCTUATION_ALIGNED_PATH = (
    PTB_TREES_ROOT_DIR + "test/test_trees_without_punct.txt"
)

YOON_KIM_TRAIN_GOLD_WITHOUT_PUNCTUATION_PATH = (
    PTB_TREES_ROOT_DIR + "Yoon_Kim/ptb-train-gold-filtered.txt"
)
YOON_KIM_VALID_GOLD_WITHOUT_PUNCTUATION_PATH = (
    PTB_TREES_ROOT_DIR + "Yoon_Kim/ptb-valid-gold-filtered.txt"
)
YOON_KIM_TEST_GOLD_WITHOUT_PUNCTUATION_PATH = (
    PTB_TREES_ROOT_DIR + "Yoon_Kim/ptb-test-gold-filtered.txt"
)

# Predictions
PTB_SAVE_TREES_PATH = "/home/wolfingten/projects/outputs/ws_parser/ger/"

# Training
TRAINED_MODEL_PATH = "/home/wolfingten/projects/models/ws_parser/ger/"
