import pycnn
import random
from collections import Counter
import util
import numpy as np

# format of files: each line is "word<TAB>tag<newline>", blank line is new sentence.
train_file = "/home/yogo/Vork/Research/corpora/pos/WSJ.TRAIN"
test_file = "/home/yogo/Vork/Research/corpora/pos/WSJ.TEST"

MLP = True


def read(fname):
    sent = []
    for line in open(fname):
        line = line.strip().split()
        if not line:
            if sent:
                yield sent
            sent = []
        else:
            word, pos_tag = line
            sent.append((word, pos_tag))


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


def build_tagging_graph(words, tags, model, builders):
    """
    build the computational graph
    :param words: list of indices
    :param tags: list of indices
    :param model: current model to access parameters
    :param builders: builder to create state combinations
    :return: joint error
    """
    pycnn.renew_cg()
    f_init, b_init = [b.initial_state() for b in builders]

    # retrieve embeddings from the model and add noise
    word_embeddings = [pycnn.lookup(model["lookup"], w) for w in words]
    word_embeddings = [pycnn.noise(we, 0.1) for we in word_embeddings]

    # compute the expressions for the forward and backward pass
    forward_sequence = [x.output() for x in f_init.add_inputs(word_embeddings)]
    backward_sequence = [x.output() for x in b_init.add_inputs(reversed(word_embeddings))]

    # retrieve model parameters
    if MLP:
        H = pycnn.parameter(pH)
        O = pycnn.parameter(pO)
    else:
        O = pycnn.parameter(pO)

    # compute per-token error
    errs = []
    for f, b, t in zip(forward_sequence, reversed(backward_sequence), tags):
        f_b = pycnn.concatenate([f, b])
        if MLP:
            # TODO: add bias terms
            r_t = O * (pycnn.tanh(H * f_b))
        else:
            r_t = O * f_b
        err = pycnn.pickneglogsoftmax(r_t, t)
        errs.append(err)

    return pycnn.esum(errs)


def tag_sent(sent, model, builders):
    # TODO: this code replicates most of the previous function, since it builds the same graph, only with a different loss/output node
    pycnn.renew_cg()
    f_init, b_init = [b.initial_state() for b in builders]
    wembs = [pycnn.lookup(model["lookup"], vw.w2i.get(w, UNK)) for w, t in sent]

    fw = [x.output() for x in f_init.add_inputs(wembs)]
    bw = [x.output() for x in b_init.add_inputs(reversed(wembs))]

    if MLP:
        H = pycnn.parameter(pH)
        O = pycnn.parameter(pO)
    else:
        O = pycnn.parameter(pO)
    tags = []
    for f, b, (w, t) in zip(fw, reversed(bw), sent):
        if MLP:
            r_t = O * (pycnn.tanh(H * pycnn.concatenate([f, b])))
        else:
            r_t = O * pycnn.concatenate([f, b])
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
                tags = tag_sent(sent, model, builders)
                golds = [t for w, t in sent]
                for go, gu in zip(golds, tags):
                    if go == gu:
                        good += 1
                    else:
                        bad += 1
            print(good / (good + bad))
        ws = [vw.w2i.get(w, UNK) for w, p in s]
        ps = [vt.w2i[p] for w, p in s]
        sum_errs = build_tagging_graph(ws, ps, model, builders)

        total_loss += sum_errs.scalar_value()
        tagged += len(ps)
        sum_errs.backward()
        sgd.update()
