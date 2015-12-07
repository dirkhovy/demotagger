import pycnn
import random
from collections import Counter
import util
import numpy as np
import argparse

parser = argparse.ArgumentParser(
    description="train a demographics predictor. File formats: CoNLL, plus one line with CSV demographic values")
parser.add_argument('--train', help='train files')
parser.add_argument('--test', help='test files')
parser.add_argument('--target', help='predict age, gender, both of them, or the joint cross-product', choices=('age', 'gender', 'both', 'joint'), default='both')
args = parser.parse_args()

# read in data
train_file = args.train
test_file = args.test

MLP = True


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
    labels = []
    for line in open(fname):
        line = line.strip().split()

        if not line:
            if sentence:
                yield (sentence, labels)
            sentence = []
            labels = []

        else:
            elements = line.split('\t')

            # read in age and gender info
            if len(elements) == 1:
                age, gender = line.split(',')
                if args.target == 'age':
                    labels = [age]
                elif args.target == 'gender':
                    labels = [gender]
                elif args.target == 'both':
                    labels = [age, gender]
                elif args.target == 'joint':
                    labels = ['%s-%s' % (age, gender)]

            # read in words and tags
            elif len(elements) == 2:
                word, pos_tag = elements
                sentence.append((word, pos_tag))

            else:
                print('Problem reading input file "%s": unexpected line "%s"' % (fname, line))


train = list(read(train_file))
test = list(read(test_file))

words = []
tags = []
wc = Counter()
for sentence in train:
    for word_sequence, pos_sequence in sentence:
        words.append(word_sequence)
        tags.append(pos_sequence)

wc.update(words)
words.append("_UNK_")

vw = util.Vocab.from_corpus([words])
vt = util.Vocab.from_corpus([tags])
UNK = vw.w2i["_UNK_"]

num_words = vw.size()
num_tags = vt.size()

model = pycnn.Model()
sgd = pycnn.SimpleSGDTrainer(model)

# TODO: replace numbers with constants or arguments
model.add_lookup_parameters("lookup", (num_words, 128))
model.add_lookup_parameters("tl", (num_tags, 30))
if MLP:
    pH = model.add_parameters("HID", (32, 50 * 2))
    pO = model.add_parameters("OUT", (num_tags, 32))
else:
    pO = model.add_parameters("OUT", (num_tags, 50 * 2))

builders = [
    pycnn.LSTMBuilder(1, 128, 50, model),
    pycnn.LSTMBuilder(1, 128, 50, model),
]


def build_tagging_graph(words, model, builders):
    """
    build the computational graph
    :param words: list of indices
    :param tags: list of indices
    :param model: current model to access parameters
    :param builders: builder to create state combinations
    :return: forward and backward sequence, plus tags and labels
    """
    pycnn.renew_cg()
    f_init, b_init = [b.initial_state() for b in builders]

    # retrieve embeddings from the model and add noise
    word_embeddings = [pycnn.lookup(model["lookup"], w) for w in words]
    word_embeddings = [pycnn.noise(we, 0.1) for we in word_embeddings]

    # compute the expressions for the forward and backward pass
    forward_sequence = [x.output() for x in f_init.add_inputs(word_embeddings)]
    backward_sequence = [x.output() for x in b_init.add_inputs(reversed(word_embeddings))]

    return list(zip(forward_sequence, reversed(backward_sequence)))


def fit(words, tags, labels, model, builders):
    """
    compute joint error of the
    :param words:
    :param tags:
    :param labels:
    :param model:
    :param builders:
    :return: joint error
    """
    # retrieve model parameters
    if MLP:
        H = pycnn.parameter(pH)
        O = pycnn.parameter(pO)
    else:
        O = pycnn.parameter(pO)

    errs = []
    for (forward_state, backward_state), tag in zip(build_tagging_graph(words, model, builders), tags):
        f_b = pycnn.concatenate([forward_state, backward_state])
        if MLP:
            # TODO: add bias terms
            r_t = O * (pycnn.tanh(H * f_b))
        else:
            r_t = O * f_b
        err = pycnn.pickneglogsoftmax(r_t, tag)
        errs.append(err)

    return pycnn.esum(errs)


def predict(sent, model, builders):
    """
    predict tags and demographic labels
    :param sent:
    :param model:
    :param builders:
    :return:
    """
    if MLP:
        H = pycnn.parameter(pH)
        O = pycnn.parameter(pO)
    else:
        O = pycnn.parameter(pO)

    tags = []
    for forward_state, backward_state in build_tagging_graph(words, model, builders):
        if MLP:
            r_t = O * (pycnn.tanh(H * pycnn.concatenate([forward_state, backward_state])))
        else:
            r_t = O * pycnn.concatenate([forward_state, backward_state])
            
        out = pycnn.softmax(r_t)
        chosen = np.argmax(out.npvalue())
        tags.append(vt.i2w[chosen])

    return tags


tagged = total_loss = 0
for ITER in range(50):
    random.shuffle(train)
    for i, s in enumerate(train, 1):
        if i % 5000 == 0:
            sgd.status()
            print(total_loss / tagged)
            total_loss = 0
            tagged = 0
        if i % 10000 == 0:
            good = bad = 0.0
            for sent in test:
                tags = predict(sent, model, builders)
                golds = [t for w, t in sent]
                for go, gu in zip(golds, tags):
                    if go == gu:
                        good += 1
                    else:
                        bad += 1
            print(good / (good + bad))
        ws = [vw.w2i.get(w, UNK) for w, p in s]
        ps = [vt.w2i[p] for w, p in s]
        sum_errs = fit(ws, ps, model, builders)

        total_loss += sum_errs.scalar_value()
        tagged += len(ps)
        sum_errs.backward()
        sgd.update()
