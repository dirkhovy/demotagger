import argparse
import numpy as np

from pathlib import Path

import re


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


parser = argparse.ArgumentParser(description="Converts a polarity dataset to the demographic format")
parser.add_argument('positive_file', type=Path)
parser.add_argument('negative_file', type=Path)
parser.add_argument('out_path', type=Path)
args = parser.parse_args()

pos_sentences = [clean_str(line).strip().split(" ") for line in args.positive_file.open(encoding='iso-8859-1')]
neg_sentences = [clean_str(line).strip().split(" ") for line in args.negative_file.open(encoding='iso-8859-1')]

labels = ['POS'] * len(pos_sentences) + ['NEG'] * len(neg_sentences)
sentences = pos_sentences + neg_sentences

indices = np.arange(len(sentences))
np.random.shuffle(indices)
datasets = [('train', indices[:7500]), ('dev', indices[7500:8750]), ('test', indices[8750:])]

for dataset_name, dataset_indices in datasets:
    out_filename = args.out_path / "rt_polarity.{}".format(dataset_name)
    with out_filename.open("w") as out_file:
        for i in dataset_indices:
            for token in sentences[i]:
                if len(token.strip()):
                    print("{}\t_".format(token.strip()), file=out_file)
            print("NA,{}".format(labels[i]), file=out_file)
            print(file=out_file)
