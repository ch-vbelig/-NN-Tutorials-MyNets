import io
import os
import unicodedata
import string
import glob
import torch
import random

DATA_DIR = 'data/names/*.txt'

# Alphabet small & capital letters & ".,;"
# abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,;
ALL_LETTERS = string.ascii_letters + '.,;'
N_LETTERS = len(ALL_LETTERS)
print(N_LETTERS)

# Convert unicode string to ascii string
# Hosé -> Hose
def unicode_to_ascii(s):
    chars = [c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn' and c in ALL_LETTERS]
    return "".join(chars)


def load_data():
    category_lines = {}
    all_categories = []

    def find_files(path):
        return glob.glob(path)

    # Read a file and split into lines
    def read_lines(filename):
        lines_original = io.open(filename, encoding='utf-8').read().strip().split('\n')
        lines_ascii = [unicode_to_ascii(s) for s in lines_original]
        return lines_ascii

    for filename in find_files(DATA_DIR):
        category = os.path.splitext(os.path.basename(filename))[0]
        category_lines[category] = read_lines(filename)

        all_categories.append(category)

    return category_lines, all_categories

"""
To represent a single letter, one-hot vector is used of size (1, n_letters)
To make a word, we join a bunch of one-hot encoded vectors into 2D matrix (line_length, n_letters)
We add 1 dimension because Pytorch assumes everything is in batches -> (line_length, 1, n_letters)
"""

# find letter index from all_letters
def letter_to_index(letter):
    index = ALL_LETTERS.find(letter)
    return index

# Turn a line into a (line_length, 1, n_letters) vector
def line_to_tensor(line):
    tensor = torch.zeros(len(line), 1, N_LETTERS)

    for i, letter in enumerate(line):
        letter_index = letter_to_index(letter)
        tensor[i][0][letter_index] = 1

    return tensor

def random_training_example(category_lines, all_categories):

    def random_choice(a):
        random_inx = random.randint(0, len(a) - 1)
        return a[random_inx]

    category = random_choice(all_categories)
    line = random_choice(category_lines[category])

    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long) # [[1], [2] ...]
    line_tensor = line_to_tensor(line)

    return category, line, category_tensor, line_tensor

if __name__ == '__main__':
    print(ALL_LETTERS)
    print(unicode_to_ascii("Hosé"))

    category_lines, all_categories = load_data()
    print(category_lines["Italian"][:4])

    print(line_to_tensor('Jones').size())




load_data()