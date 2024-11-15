import re
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import json
import config

TEXT_PATH = config.TEXT_PATH
TRAIN_DICT_PATH = config.TRAIN_DICT_PATH
VAL_DICT_PATH = config.VAL_DICT_PATH


def open_file(fpath):
    with open(fpath) as f:
        text = f.read()
    return text


def text_cleaner(text_in):
    # 0. lower case
    text = text_in.lower()

    # remove punctuations
    text = re.sub("[0-9!?,./\-:;«»()—]", "", text)
    text = re.sub("\n", "", text)
    text = re.sub("\s{2}", "", text)
    text = re.sub("\s{2}", "", text)

    return text

def generate_sequences(text, seq_length=30):
    sequences = []
    for i in range(0, len(text) - seq_length):
        # select window of seq_length
        seq = text[i:i + seq_length]

        # store
        sequences.append(seq)

    print(f'Total sequences: {len(sequences)}')
    return sequences


def get_encoder_decoder(text):
    # unique characters sorted
    chars = sorted(set(list(text)))

    char_to_idx = dict(((c, i) for i, c in enumerate(chars)))
    idx_to_char = dict(((i, c) for i, c in enumerate(chars)))

    return char_to_idx, idx_to_char

def encode_sequence(sequences: list, char_to_idx: dict):
    encoded_sequences = []

    for seq in sequences:
        # encode into integers each line
        encoded = [char_to_idx[c] for c in seq]
        # store encoded
        encoded_sequences.append(encoded)

    return encoded_sequences


def get_train_and_test_data(encoded_sequences, char_to_idx):
    # vocabulary size
    vocab_size = len(char_to_idx)

    sequences = np.array(encoded_sequences)

    #
    x, y = sequences[:, :-1], sequences[:, -1]

    # # one hot x
    # x = to_categorical(x, num_classes=vocab_size)

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1, random_state=30)

    print(f'Train shape: {x_train.shape}, Val shape: {x_val.shape}')

    return x_train, x_val, y_train, y_val


def convert_to_dict(x: np.array, y, char_to_idx, idx_to_char):
    data = {
        'char_to_idx': {},
        'idx_to_char': {},
        'data': [],
        'targets': []
    }

    data['char_to_idx'] = char_to_idx
    data['idx_to_char'] = idx_to_char
    data['data'] = x.tolist()
    data['targets'] = y.tolist()

    return data


if __name__ == '__main__':

    # open raw text
    text = open_file(TEXT_PATH)

    # clean text
    text = text_cleaner(text)

    # get sequences
    sequences = generate_sequences(text)

    # get encoder and decoder
    char_to_idx, idx_to_char = get_encoder_decoder(text)

    # encode
    encoded_sequences = encode_sequence(sequences, char_to_idx)

    # train and test split
    x_train, x_val, y_train, y_val = get_train_and_test_data(encoded_sequences, char_to_idx)

    #
    train_dict = convert_to_dict(x_train, y_train, char_to_idx, idx_to_char)
    val_dict = convert_to_dict(x_val, y_val, char_to_idx, idx_to_char)

    # Save train and validation data in json
    with open(TRAIN_DICT_PATH, 'w') as fp:
        json.dump(train_dict, fp)

    with open(VAL_DICT_PATH, 'w') as fp:
        json.dump(val_dict, fp)









