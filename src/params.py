LABELS = {
    'earthquake': 0,
    'typhoon': 1,
    'biological': 2,
    'flood': 3,
}

NUM_LAYERS = 7 
LEARNING_RATE = 0.1
DROPOUT = 0.2
NUM_CLASSES = 4
EPOCHS = 10 
BATCH_SIZE = 158

DATA_PATH = '../data_set/'

# Vectorization parameters
# Range (inclusive) of n-gram sizes for tokenizing text.
NGRAM_RANGE = (1, 2)

# Limit on the number of features. We use the top 20K features.
TOP_K = 20000

# Whether text should be split into word or character n-grams.
# One of 'word', 'char'.
TOKEN_MODE = 'word'

# Minimum document/corpus frequency below which a token will be discarded.
MIN_DOCUMENT_FREQUENCY = 2