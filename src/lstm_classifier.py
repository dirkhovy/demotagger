import argparse
import pycnn
import random
import sys
import time

import numpy as np
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


def read(fname, target):
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
    include = True

    for line in open(fname):
        line = line.strip()

        if not line:
            if sentence != [] and include:
                yield (sentence, labels)
            sentence = []
            labels = None
            include = True

        else:
            elements = line.split('\t')

            # read in age and gender info
            if len(elements) == 1:
                age, gender = line.split(',')

                if target == 'age':
                    if age == 'NONE':
                        include = False
                    labels = age
                elif target == 'gender':
                    if gender == 'NONE':
                        include = False
                    labels = gender
                elif target == 'joint':
                    labels = '%s-%s' % (age, gender)
                    if age == 'NONE' or gender == 'NONE':
                        include = False
                elif target == 'both':
                    labels = [age if age != 'NONE' else None, gender if gender != 'NONE' else None]

            # read in words and tags
            elif len(elements) == 2:
                word, pos_tag = elements
                sentence.append(word)

            else:
                print('Problem reading input file "%s": unexpected line "%s"' % (fname, line))


train = list(read(train_file, args.target))
test = list(read(test_file, args.target))
dev = list(read(dev_file, args.target))

words = set()
labels = set()
age_labels = set()
gender_labels = set()

for sentence, label in train:
    words.update(sentence)
    if args.target == 'both':
        for i, individual_label in enumerate(label):
            if label is not None:
                labels.add(individual_label)
                if i == 0:
                    age_labels.add(individual_label)
                else:
                    gender_labels.add(individual_label)

    else:
        labels.add(label)
        if args.target == 'age':
            age_labels.add(label)
        elif args.target == 'gender':
            gender_labels.add(label)

try:
    age_labels.remove(None)
    gender_labels.remove(None)
except KeyError:
    print('No unspecified labels in target', file=sys.stderr)

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
i2al = list(age_labels)
al2i = {label: i for i, label in enumerate(i2al)}
i2gl = list(gender_labels)
gl2i = {label: i for i, label in enumerate(i2gl)}

UNK = w2i["_UNK_"]
EOS = w2i["_EOS_"]


def convert_instances(data_set):
    oov = 0
    for i, (sentence, label) in enumerate(data_set):
        word_indices = [w2i.get(word, UNK) for word in sentence]
        oov += word_indices.count(UNK)
        word_indices.append(EOS)
        # word_indices = np.array(word_indices)
        if args.target == 'both':
            label_index = [al2i[label[0]] if label[0] is not None else None,
                           gl2i[label[1]] if label[1] is not None else None]
        else:
            label_index = l2i[label]
        data_set[i] = (word_indices, label_index)
    return data_set, oov


# convert words to word_indices
train, oov_train = convert_instances(train)
test, oov_test = convert_instances(test)
dev, oov_dev = convert_instances(dev)

# for both, split data into labels, add target type to each instance
if args.target == 'both':
    train_age = [((words, labels[0]), 'age') for words, labels in train if labels[0] is not None]
    train_gender = [((words, labels[1]), 'gender') for words, labels in train if labels[1] is not None]
    smaller_training_size = min(len(train_age), len(train_gender))
    print("age/gender-split: %s to %s " % (len(train_age), len(train_gender)), smaller_training_size)
# otherwise, just add target type to each instance
else:
    train = [(y, args.target) for y in train]

num_words = len(w2i)
num_labels = len(l2i)

print("read in all the files!\n", '*' * 20, file=sys.stderr)
print("%s train instances (%s OOV words), %s test instances (%s OOV words), %s dev instances (%s OOV words)" % (
    len(train), oov_train, len(test), oov_test, len(dev), oov_dev), file=sys.stderr)
print("%s words, %s labels (%s)" % (num_words, num_labels, i2l), file=sys.stderr)
if args.target == 'both':
    print("\tage labels: %s, gender labels: %s" % (i2al, i2gl), file=sys.stderr)

model = pycnn.Model()
trainers = {'sgd': pycnn.SimpleSGDTrainer, 'adam': pycnn.AdamTrainer, 'adagrad': pycnn.AdagradTrainer,
            'adadelta': pycnn.AdadeltaTrainer, 'momentum': pycnn.MomentumSGDTrainer}
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

if args.target in ['joint']:
    pOutAge = model.add_parameters("OUT_AGE", (num_labels, MLP_HIDDEN_LAYER_SIZE / 2))
    biasOutAge = model.add_parameters("BIAS_OUT_AGE", (num_labels))
elif args.target in ['age', 'both']:
    pOutAge = model.add_parameters("OUT_AGE", (len(age_labels), MLP_HIDDEN_LAYER_SIZE / 2))
    biasOutAge = model.add_parameters("BIAS_OUT_AGE", (len(age_labels)))

if args.target in ['gender', 'both']:
    pOutGender = model.add_parameters("OUT2", (len(gender_labels), MLP_HIDDEN_LAYER_SIZE / 2))
    biasOutGender = model.add_parameters("BIAS_OUT2", (len(gender_labels)))

# TODO: do we need a second hidden layer, or do we want to pool information there?
pH2 = model.add_parameters("HID2", (MLP_HIDDEN_LAYER_SIZE / 2, MLP_HIDDEN_LAYER_SIZE))
biasH2 = model.add_parameters("BIAS_HIDDEN2", (MLP_HIDDEN_LAYER_SIZE / 2))

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
def fit(word_indices, label, model, builder, target):
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
    bias_H = pycnn.parameter(biasH)
    H2 = pycnn.parameter(pH2)
    bias_H2 = pycnn.parameter(biasH2)

    if target in ['age', 'joint']:
        O = pycnn.parameter(pOutAge)
        bias_O = pycnn.parameter(biasOutAge)
    elif target == 'gender':
        O = pycnn.parameter(pOutGender)
        bias_O = pycnn.parameter(biasOutGender)

    # print(pycnn.cg().PrintGraphviz())
    # if target == 'both':
    #     hidden = bias_H + pycnn.tanh(H * final_state)
    #     r_age = bias_O + (O * hidden)
    #     r_gender = bias_O2 + (O2 * hidden)
    #     return pycnn.esum([pycnn.pickneglogsoftmax(r_age, label[0]), pycnn.pickneglogsoftmax(r_gender, label[1])])

    r_t = bias_O + (O * (bias_H2 + pycnn.tanh(H2 * (bias_H + pycnn.tanh(H * final_state)))))
    return pycnn.pickneglogsoftmax(r_t, label)


def predict(word_indices, model, builder, target):
    """
    predict demographic label
    :param word_indices:
    :param model:
    :param builder:
    :return: tag and label predictions
    """
    forward_states = build_tagging_graph(word_indices, model, builder)
    final_state = forward_states[-1]

    H = pycnn.parameter(pH)
    bias_H = pycnn.parameter(biasH)
    H2 = pycnn.parameter(pH2)
    bias_H2 = pycnn.parameter(biasH2)

    if target in ['age', 'both', 'joint']:
        O = pycnn.parameter(pOutAge)
        bias_O = pycnn.parameter(biasOutAge)
    elif target == 'gender':
        O = pycnn.parameter(pOutGender)
        bias_O = pycnn.parameter(biasOutGender)

    if target == 'both':
        O2 = pycnn.parameter(pOutGender)
        bias_O2 = pycnn.parameter(biasOutGender)


    if target == 'both':
        hidden = bias_H2 + pycnn.tanh(H2 * (bias_H + pycnn.tanh(H * final_state)))
        r_age = bias_O + (O * hidden)
        r_gender = bias_O2 + (O2 * hidden)

        out_age = pycnn.softmax(r_age)
        out_gender = pycnn.softmax(r_gender)

        return [np.argmax(out_age.npvalue()), np.argmax(out_gender.npvalue())]

    else:
        r_t = bias_O + (O * (bias_H2 + pycnn.tanh(H2 * (bias_H + pycnn.tanh(H * final_state)))))

        out = pycnn.softmax(r_t)
        chosen = np.argmax(out.npvalue())

        return chosen


def evaluate(data_set, model, builder, target):
    """
    evaluate a test file
    :param data_set: the converted input file, i.e., a list of ([word_indices], label_index)
    :param model: current model
    :param builder: current builder
    :return: per-token accuracy
    """
    good = 0.0
    bad = 0.0

    individual_good = [0.0, 0.0]
    individual_bad = [0.0, 0.0]

    for test_sentence, test_label in data_set:
        predicted_label = predict(test_sentence, model, builder, target)

        if predicted_label == test_label:
            good += 1
        else:
            bad += 1

        if target == 'both':
            label_dict = {0: i2al, 1: i2gl}
            for i, (prediction, gold) in enumerate(zip(predicted_label, test_label)):
                if prediction == gold:
                    individual_good[i] += 1
                else:
                    # print('\tMISTAKE:', label_dict[i][prediction], label_dict[i][gold], file=sys.stderr)
                    individual_bad[i] += 1

    if target == 'both':
        return "%.4f instance accuracy, label accuracy %.4f age, %.4f gender" % (
            good / (good + bad), individual_good[0] / (individual_good[0] + individual_bad[0]),
            individual_good[1] / (individual_good[1] + individual_bad[1]))
    else:
        return '%.4f' % (good / (good + bad))


num_instances = 1
losses = []
start = time.time()

current_best_dev = 0.0
current_best_test = 0.0
best_model = 0

for iteration in range(args.iterations):

    # re-sample both target types and add their
    if args.target == 'both':
        train = random.sample(train_age, smaller_training_size)
        train.extend(random.sample(train_gender, smaller_training_size))

    random.shuffle(train)
    batch_i = 0

    for i, ((sentence, label), instance_target) in enumerate(train, 1):

        if num_instances % args.status == 0:
            print('ITERATION %s, instance %s/%s' % (iteration + 1, i, len(train)), file=sys.stderr, end='\t')
            print('Avg. loss', sum(losses) / len(losses), file=sys.stderr, end='\t')
            print('time: %.2f sec' % (time.time() - start), file=sys.stderr)
            # sgd.status()
            start = time.time()
            losses.clear()

        if num_instances % (args.status * 5) == 0:
            print('-' * 50, file=sys.stderr)
            print("Accuracy on dev: %s\n" % (evaluate(dev, model, builder, args.target)), file=sys.stderr)

        # TRAINING
        error = fit(sentence, label, model, builder, instance_target)

        losses.append(error.scalar_value())
        error.backward()

        num_instances += 1
        batch_i += 1

        if batch_i == args.batch:
            batch_i = 0
            sgd.update()

    print('=' * 50, file=sys.stderr)
    dev_accuracy = evaluate(dev, model, builder, args.target)
    test_accuracy = evaluate(test, model, builder, args.target)
    if float(dev_accuracy.split()[0]) > current_best_dev:
        current_best_dev = float(dev_accuracy.split()[0])
        current_best_test = float(test_accuracy.split()[0])
        best_model = iteration + 1
    print("iteration %s. Accuracy on dev: %s (best: %s @ %s)" % (iteration + 1, dev_accuracy, current_best_dev, best_model))
    print("iteration %s. Accuracy on test: %s (best: %s @ %s)" % (iteration + 1, test_accuracy, current_best_test, best_model))
    print('iteration %s. Avg. loss %s' % (iteration + 1, sum(losses) / len(losses)))
    print('=' * 50, file=sys.stderr)
    print('', file=sys.stderr)
    num_instances = 1
    start = time.time()
    losses.clear()
