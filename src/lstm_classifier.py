import argparse
import pycnn
import random
import sys
import numpy as np
import time
from numba import jit

parser = argparse.ArgumentParser(
    description="train a demographics predictor. File formats: CoNLL, plus one line with CSV demographic values")
parser.add_argument('--train', help='train files', required=True)
parser.add_argument('--dev', help='dev files', required=True)
parser.add_argument('--test', help='test files', required=True)
parser.add_argument('--embeddings', help='embeddings files (word2vec format)', default=None)
parser.add_argument('--target', help='predict age, gender, both of them, or the joint cross-product',
                    choices=('age', 'gender', 'both', 'joint'), default='both')
parser.add_argument('--trainer', help='which training algorithm to use',
                    choices=('adagrad', 'sgd', 'adam', 'adadelta', 'momentum'), default='adam')
parser.add_argument('--iterations', help='number of iterations', default=50)
parser.add_argument('--status', help='number of processed instances between status updates', default=10000, type=int)
parser.add_argument('--noise', help='amount of noise added to embeddings', default=0.1, type=float)
parser.add_argument('--n_embeddings', help='dimensionality of word embeddings', default=128, type=int)
parser.add_argument('--n_hidden', help='dimensionality of hidden LSTM layer', default=50, type=int)
parser.add_argument('--n_output', help='dimensionality of output layer', default=32, type=int)
parser.add_argument('--dropout', help='dropout probability for final sentence representation', default=0.0, type=float)
parser.add_argument('--batch', help='batch size', default=1, type=int)
args = parser.parse_args()

# read in data
train_file = args.train
test_file = args.test
dev_file = args.dev


def read_embeddings_file(file_name):
    """
    read in a file with space separated embeddings of the format
    <WORD Value1 Value2 ... ValueN>
    :param file_name: embeddings file
    :return: dictionary {word: [float(values)]}
    """
    words = []
    embeddings = []
    for line in open(file_name):
        elements = line.strip().split(' ')
        if len(elements) > 2:
            words.append(elements[0])
            embeddings.append(list(map(float, elements[1:])))

    return words, embeddings


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
                elif args.target == 'both':
                    labels = [age, gender]
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
dev = list(read(dev_file))

words = set()
labels = set()

for sentence, label in train:
    words.update(sentence)
    if args.target == 'both':
        for individual_label in label:
            labels.add(individual_label)
    else:
        labels.add(label)

# id we have pre-trained embeddings, read them in (preserving order), and add the special tokens
if args.embeddings:
    print('reading embeddings file', file=sys.stderr)
    words, pre_trained_embeddings = read_embeddings_file(args.embeddings)
    embedding_dimensionality = len(pre_trained_embeddings[0])
    words.append("_UNK_")
    pre_trained_embeddings.append([0.0] * embedding_dimensionality)

    words.append("_EOS_")
    pre_trained_embeddings.append([1.0] * embedding_dimensionality)

# otherwise, just add the special tokens (order does not matter)
else:
    words.add("_UNK_")
    words.add("_EOS_")

# get the hashing/bookkeeping objects
i2w = list(words)
w2i = {word: i for i, word in enumerate(i2w)}
i2l = list(labels)
l2i = {label: i for i, label in enumerate(i2l)}
UNK = w2i["_UNK_"]
EOS = w2i["_EOS_"]

# convert words to word_indices
oov_train = 0
oov_test = 0
oov_dev = 0
for i, (sentence, label) in enumerate(train):
    word_indices = [w2i.get(word, UNK) for word in sentence]
    oov_train += word_indices.count(UNK)
    word_indices.append(EOS)
    # word_indices = np.array(word_indices)
    label_index = l2i[label]
    train[i] = (word_indices, label_index)

for i, (sentence, label) in enumerate(test):
    word_indices = [w2i.get(word, UNK) for word in sentence]
    oov_test += word_indices.count(UNK)
    word_indices.append(EOS)
    # word_indices = np.array(word_indices)
    label_index = l2i[label]
    test[i] = (word_indices, label_index)

for i, (sentence, label) in enumerate(dev):
    word_indices = [w2i.get(word, UNK) for word in sentence]
    oov_dev += word_indices.count(UNK)
    word_indices.append(EOS)
    # word_indices = np.array(word_indices)
    label_index = l2i[label]
    dev[i] = (word_indices, label_index)

num_words = len(w2i)
num_labels = len(l2i)

print("read in all the files!\n", '*' * 20, file=sys.stderr)
print("%s train instances (%s OOV words), %s test instances (%s OOV words), %s dev instances (%s OOV words)" % (
len(train), oov_train, len(test), oov_test, len(dev), oov_dev), file=sys.stderr)
print("%s words, %s labels\n" % (num_words, num_labels), file=sys.stderr)

model = pycnn.Model()
trainers = {'sgd': pycnn.SimpleSGDTrainer, 'adam': pycnn.AdamTrainer, 'adagrad': pycnn.AdagradTrainer, 'adadelta': pycnn.AdadeltaTrainer, 'momentum': pycnn.MomentumSGDTrainer}
sgd = trainers[args.trainer](model)

print("declared model and trainer (%s)" % (args.trainer), file=sys.stderr)

WORD_EMBEDDING_SIZE = args.n_embeddings
LSTM_HIDDEN_LAYER_SIZE = args.n_hidden
MLP_HIDDEN_LAYER_SIZE = args.n_output

word_embeddings = model.add_lookup_parameters("word_lookup", (num_words, WORD_EMBEDDING_SIZE))
if args.embeddings:
    word_embeddings.init_from_array(np.array(pre_trained_embeddings))

pH = model.add_parameters("HID", (MLP_HIDDEN_LAYER_SIZE, LSTM_HIDDEN_LAYER_SIZE))
biasH = model.add_parameters("BIAS_HIDDEN", (MLP_HIDDEN_LAYER_SIZE))
pO = model.add_parameters("OUT", (num_labels, MLP_HIDDEN_LAYER_SIZE))
biasO = model.add_parameters("BIAS_OUT", (num_labels))

if args.target == 'both':
    pH2 = model.add_parameters("HID2", (MLP_HIDDEN_LAYER_SIZE, LSTM_HIDDEN_LAYER_SIZE))
    biasH2 = model.add_parameters("BIAS_HIDDEN2", (MLP_HIDDEN_LAYER_SIZE))
    pO2 = model.add_parameters("OUT2", (num_labels, MLP_HIDDEN_LAYER_SIZE))
    biasO2 = model.add_parameters("BIAS_OUT2", (num_labels))

print("declared variables", file=sys.stderr)

builder = pycnn.LSTMBuilder(1, WORD_EMBEDDING_SIZE, LSTM_HIDDEN_LAYER_SIZE, model)
# builder = pycnn.SimpleRNNBuilder(1, WORD_EMBEDDING_SIZE, LSTM_HIDDEN_LAYER_SIZE, model)
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
    word_embeddings = [pycnn.noise(we, args.noise) for we in word_embeddings]

    # compute the expressions for the forward pass
    forward_sequence = [x.output() for x in f_init.add_inputs(word_embeddings)]

    return forward_sequence

@jit
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
    final_state = pycnn.dropout(final_state, 0.1)
    # print("final state", final_state, file=sys.stderr)
    H = pycnn.parameter(pH)
    O = pycnn.parameter(pO)
    bias_O = pycnn.parameter(biasO)
    bias_H = pycnn.parameter(biasH)

    # print(pycnn.cg().PrintGraphviz())

    # TODO: add bias terms
    r_t = bias_O + (O * (bias_H + pycnn.tanh(H * final_state)))

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
    bias_O = pycnn.parameter(biasO)
    bias_H = pycnn.parameter(biasH)

    final_state = forward_states[-1]

    # TODO: add bias terms
    r_t = bias_O + (O * (bias_H + pycnn.tanh(H * final_state)))

    out = pycnn.softmax(r_t)
    chosen = np.argmax(out.npvalue())

    return chosen


def evaluate(data_set, model, builder):
    """
    evaluate a test file
    :param data_set: the converted input file, i.e., a list of ([word_indices], label_index)
    :param model: current model
    :param builder: current builder
    :return: per-token accuracy
    """
    good = 0.0
    bad = 0.0

    for test_sentence, test_label in data_set:
        predicted_label = predict(test_sentence, model, builder)

        if predicted_label == test_label:
            good += 1
        else:
            bad += 1

    return good / (good + bad)


num_instances = 1
losses = []
start = time.time()

for iteration in range(args.iterations):
    random.shuffle(train)
    batch_i = 0

    for i, (sentence, label) in enumerate(train, 1):

        if num_instances % args.status == 0:
            print('ITERATION %s, instance %s/%s' % (iteration + 1, i, len(train)), file=sys.stderr, end='\t')
            # sgd.status()
            print('Avg. loss', sum(losses) / len(losses), file=sys.stderr, end='\t')
            print('time: %.2f sec' % (time.time() - start), file=sys.stderr)
            start = time.time()
            losses.clear()

        if num_instances % (args.status * 5) == 0:
            print('-' * 50, file=sys.stderr)
            print("Accuracy on dev: %s\n" % (evaluate(dev, model, builder)), file=sys.stderr)

        # TRAINING
        # fit the shit!
        error = fit(sentence, label, model, builder)

        losses.append(error.scalar_value())
        error.backward()

        num_instances += 1
        batch_i += 1

        if batch_i == args.batch:
            batch_i = 0
            sgd.update()

    print('=' * 50, file=sys.stderr)
    print("iteration %s. Accuracy on dev: %s" % (iteration + 1, evaluate(dev, model, builder)))
    print("iteration %s. Accuracy on test: %s" % (iteration + 1, evaluate(test, model, builder)))
    print('iteration %s. Avg. loss %s' %(iteration + 1, sum(losses) / len(losses)))
    print('=' * 50, file=sys.stderr)
    num_instances = 1
    start = time.time()
    losses.clear()
