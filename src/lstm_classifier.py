import argparse
import pycnn
import random

import numpy as np
import sys

parser = argparse.ArgumentParser(
    description="train a demographics predictor. File formats: CoNLL, plus one line with CSV demographic values")
parser.add_argument('--train', help='train files')
parser.add_argument('--test', help='test files')
parser.add_argument('--target', help='predict age, gender, both of them, or the joint cross-product',
                    choices=('age', 'gender', 'both', 'joint'), default='both')
parser.add_argument('--iterations', help='number of iterations', default=5000)
args = parser.parse_args()

# read in data
train_file = args.train
test_file = args.test


def read(fname):
    """
    read in a CoNLL style file
    format of files: each line is
    "word<TAB>tag<newline>", followed by
    "age,gender",
    blank line is new sentence.
    :param fname: file to read
    :return: generator of ([words, tags], [labels])
    """
    sentence = []
    labels = None

    for line in open(fname):
        line = line.strip()

        if not line:
            if sentence:
                yield (sentence, labels)
            sentence = []
            labels = None

        else:
            elements = line.split('\t')

            # read in age and gender info
            if len(elements) == 1:
                age, gender = line.split(',')
                if args.target == 'age':
                    labels = age
                elif args.target == 'gender':
                    labels = gender
                # elif args.target == 'both':
                #     labels = [age, gender]
                elif args.target == 'joint':
                    labels = '%s-%s' % (age, gender)

            # read in words and tags
            elif len(elements) == 2:
                word, pos_tag = elements
                sentence.append(word)

            else:
                print('Problem reading input file "%s": unexpected line "%s"' % (fname, line))


train = list(read(train_file))
test = list(read(test_file))

words = set()
labels = set()

for sentence, label in train:
    words.update(sentence)
    labels.add(label)
words.add("_UNK_")

i2w = list(words)
w2i = {word: i for i, word in enumerate(i2w)}
i2l = list(labels)
l2i = {label: i for i, label in enumerate(i2l)}
UNK = w2i["_UNK_"]

num_words = len(w2i)
num_labels = len(l2i)

print("read in all the files!", file=sys.stderr)
print("%s train instances, %s test instances" % (len(train), len(test)), file=sys.stderr)
print("%s words, %s labels\n" % (num_words, num_labels), file=sys.stderr)

model = pycnn.Model()
sgd = pycnn.SimpleSGDTrainer(model)

print("declared model and trainer", file=sys.stderr)

WORD_EMBEDDING_SIZE = 128
LSTM_HIDDEN_LAYER_SIZE = 50
MLP_HIDDEN_LAYER_SIZE = 32

model.add_lookup_parameters("word_lookup", (num_words, WORD_EMBEDDING_SIZE))
pH = model.add_parameters("HID", (MLP_HIDDEN_LAYER_SIZE, LSTM_HIDDEN_LAYER_SIZE))
pO = model.add_parameters("OUT", (num_labels, MLP_HIDDEN_LAYER_SIZE))
print("declared variables", file=sys.stderr)

# builder = pycnn.LSTMBuilder(1, WORD_EMBEDDING_SIZE, LSTM_HIDDEN_LAYER_SIZE, model)
builder = pycnn.SimpleRNNBuilder(1, WORD_EMBEDDING_SIZE, LSTM_HIDDEN_LAYER_SIZE, model)
print("declared builder", file=sys.stderr)



def build_tagging_graph(word_indices, model, builder):
    """
    build the computational graph
    :param word_indices: list of indices
    :param model: current model to access parameters
    :param builder: builder to create state combinations
    :return: forward and backward sequence
    """
    pycnn.renew_cg()
    f_init = builder.initial_state()

    # retrieve embeddings from the model and add noise
    word_embeddings = [pycnn.lookup(model["word_lookup"], w) for w in word_indices]
    word_embeddings = [pycnn.noise(we, 0.1) for we in word_embeddings]

    # compute the expressions for the forward pass
    forward_sequence = [x.output() for x in f_init.add_inputs(word_embeddings)]

    return forward_sequence


def fit(word_indices, label, model, builder):
    """
    compute joint error of the
    :param word_indices: list of indices
    :param label: index
    :param model: current model to access parameters
    :param builder: builder to create state combinations
    :return: joint error
    """

    forward_states = build_tagging_graph(word_indices, model, builder)

    # retrieve model parameters
    final_state = forward_states[-1]
    # print("final state", final_state, file=sys.stderr)
    H = pycnn.parameter(pH)
    O = pycnn.parameter(pO)

    # print(pycnn.cg().PrintGraphviz())

    # TODO: add bias terms
    # print("multiplying stuff!", file=sys.stderr)

    r_t = O * (pycnn.tanh(H * final_state))

    return pycnn.pickneglogsoftmax(r_t, label)


def predict(word_indices, model, builder):
    """
    predict demographic label
    :param word_indices:
    :param model:
    :param builder:
    :return: tag and label predictions
    """
    forward_states = build_tagging_graph(word_indices, model, builder)

    H = pycnn.parameter(pH)
    O = pycnn.parameter(pO)

    f_b = forward_states[-1]

    # TODO: add bias terms
    r_t = O * (pycnn.tanh(H * f_b))

    out = pycnn.softmax(r_t)
    chosen = np.argmax(out.npvalue())

    return i2l[chosen]


total_loss = 0
num_instances = 1

for iteration in range(args.iterations):
    random.shuffle(train)

    if iteration > 0 and iteration % 10 == 0:
        print('\nITERATION %s' % (iteration+1), file=sys.stderr)

    for i, (sentence, label) in enumerate(train, 1):
        if num_instances % 10000 == 0:
            sgd.status()
            print('Avg. loss', total_loss / num_instances, file=sys.stderr)
            total_loss = 0
            num_instances = 0

        if num_instances % 50000 == 0:
            good = 0.0
            bad = 0.0

            for test_sentence, test_label in test:
                test_word_indices = [w2i.get(word, UNK) for word in test_sentence]
                predicted_label = predict(test_word_indices, model, builder)

                if predicted_label == test_label:
                    good += 1
                else:
                    bad += 1

            print('='*50, file=sys.stderr)
            print("Accuracy on test: %s\n" % (good / (good + bad)), file=sys.stderr)

        # TRAINING
        # convert words to word_indices
        word_indices = [w2i.get(word, UNK) for word in sentence]
        label_index = l2i[label]

        # fit the shit!
        error = fit(word_indices, label_index, model, builder)

        total_loss += error.scalar_value()
        # print(error.scalar_value(), total_loss)
        num_instances += 1
        error.backward()
        sgd.update()
