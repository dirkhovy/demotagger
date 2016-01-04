"""
run as:
THEANO_FLAGS=mode=FAST_RUN,device=cpu python src/classifier_k.py --train ~/corpora/demotagger_data/rt-polaritydata/rt_polarity.train --test ~/corpora/demotagger_data/rt-polaritydata/rt_polarity.test  --dev ~/corpora/demotagger_data/rt-polaritydata/rt_polarity.dev --target gender
"""
import argparse
from collections import deque

import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM

from reader import read_datasets, make_dense
from datetime import datetime

parser = argparse.ArgumentParser(
    description="train a demographics predictor. File formats: CoNLL, plus one line with CSV demographic values")
parser.add_argument('--train', help='train files', required=True)
parser.add_argument('--dev', help='dev files', required=True)
parser.add_argument('--test', help='test files', required=True)
parser.add_argument('--architecture', '-a', help="Model architecture", choices=['lstm', 'cnn'])
parser.add_argument('--target', help='predict age, gender, both of them, or the joint cross-product',
                    choices=('age', 'gender', 'both', 'joint'), default='both')
parser.add_argument('--num-epochs', help='Number of epochs', default=50)
parser.add_argument('--dim-rnn', help='dimensionality of hidden RNN layer', default=50, type=int)
parser.add_argument('--dim-emb', help='dimensionality of word embeddings', default=100, type=int)
parser.add_argument('--dim-out', help='dimensionality of output layer', default=32, type=int)
# parser.add_argument('--dropout', help='dropout probability for final sentence representation', default=0.0, type=float)
parser.add_argument('--batch-size', help='batch size', default=1, type=int)
parser.add_argument('--max-len', help='Maximum length of input', default=100, type=int)
args = parser.parse_args()

# Read in data, mapping words to integers
datasets = read_datasets(args.train, args.test, args.dev, "gender") # for now just gender
train_dataset = datasets['train']
dev_dataset = datasets['dev']
word_mapper = datasets['word_mapper']
label_mapper = datasets['label_mapper']


# Create dense representations of datasets
X_train, y_train = make_dense(train_dataset['sentences'], args.max_len,
                              train_dataset['labels'], len(label_mapper))
train_sent_lens = np.array([len(sent) for sent in train_dataset['sentences']])
X_dev, y_dev = make_dense(dev_dataset['sentences'], args.max_len,
                          dev_dataset['labels'], len(label_mapper))
dev_sent_lens = np.array([len(sent) for sent in dev_dataset['sentences']])

assert len(X_train) == len(y_train)
assert len(X_train) == len(train_sent_lens)


max_features = len(word_mapper)


start = datetime.now()
print("started: ", start)

# Setup model
model = Sequential()
model.add(Embedding(max_features, args.dim_emb))
model.add(LSTM(output_dim=args.dim_rnn))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='rmsprop')

model.fit(X_train, y_train, batch_size=args.batch_size, nb_epoch=args.num_epochs,
          show_accuracy=True, verbose=1, validation_data=(X_dev,y_dev))

end=datetime.now()
dur = start-end
print("ended: ", end, end-start, dur.seconds)
