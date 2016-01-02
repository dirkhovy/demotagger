import numpy as np

START_SYMBOL = "<S>"
END_SYMBOL = "</S>"
OOV_SYMBOL = "UNK"
PAD_SYMBOL = "PAD"

class FeatDict:
    def __init__(self):
        self.id_to_name = {}
        self.name_to_id = {}

    def map(self, name, frozen=False):
        assert name is not None
        feat_id = self.name_to_id.get(name)
        if feat_id is None and not frozen:
            feat_id = len(self.name_to_id)
            self.name_to_id[name] = feat_id

        return feat_id

    def update_reverse_mapping(self):
        self.id_to_name = {feat_id: feat_name for feat_name, feat_id in self.name_to_id.items()}

    def __len__(self):
        return len(self.name_to_id)



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


def read_sentences_and_labels(fname, feat_mapper, frozen=False, add_boundaries=False):
    """
    read in a CoNLL style file
    format of files: each line is
    "word<TAB>tag<newline>", followed by
    "age,gender",
    blank line is new sentence.
    :param fname: file to read
    :return: a list of sentences, a list of labels
    """
    all_sentences = []
    all_multi_labels = []
    words = [START_SYMBOL] if add_boundaries else []

    # Assume that the file ends with blank line
    for line_no, line in enumerate(open(fname), 1):
        line = line.strip()

        if len(line) == 0:
            continue

        elements = line.split('\t')

        # read in age and gender info
        if len(elements) == 1:
            age, gender = line.split(',')
            all_multi_labels.append((age, gender))
            if add_boundaries:
                words.append(END_SYMBOL)
            all_sentences.append(words)
            words = [START_SYMBOL] if add_boundaries else []

        # read in words and tags
        elif len(elements) == 2:
            word, pos_tag = elements
            words.append(feat_mapper.map(word))

        assert len(elements) <= 2, "Invalid input in file {} line {}".format(fname, line_no)

    assert len(all_sentences), len(all_multi_labels)
    return all_sentences, all_multi_labels

def get_label(label_list, target):
    age, gender = label_list
    if age and target == 'age':
        return age
    if gender and target == 'gender':
        return gender
    if age and gender and target == 'both':
        return "{}+{}".format(age, gender)

def read_datasets(train_fname, test_fname, dev_fname, target, add_boundaries=False):
    word_mapper = FeatDict()
    word_mapper.map(PAD_SYMBOL)
    label_mapper = FeatDict()

    datasets = {'word_mapper': word_mapper,
                'label_mapper': label_mapper
                }

    frozen = False
    for dataset_name, fname in [('train', train_fname), ('test', test_fname), ('dev', dev_fname)]:
        sentences, multi_labels = read_sentences_and_labels(fname, word_mapper, frozen, add_boundaries)

        filt_sentences = []
        new_labels = []
        for sent, label_list in zip(sentences, multi_labels):
            new_label = label_mapper.map(get_label(label_list, target), frozen)

            assert new_label is not None, "Label {} not seen in training".format(get_label(label_list, target))
            if new_label is not None:
                filt_sentences.append(sent)
                new_labels.append(new_label)

        datasets[dataset_name] = {'sentences': filt_sentences,
                                  'labels': new_labels,
                                  }

        if dataset_name == 'train':
            frozen = True

    word_mapper.update_reverse_mapping()
    label_mapper.update_reverse_mapping()
    return datasets


def make_dense(sentences, max_sent_len, labels, num_labels):
    assert len(sentences) == len(labels)
    X = np.zeros([len(sentences), max_sent_len])
    for i, sent in enumerate(sentences):
        trunc_sent = sent[:max_sent_len]
        X[i, :len(trunc_sent)] = trunc_sent

    y = np.zeros([len(labels), num_labels])
    for i, label in enumerate(labels):
        y[i, label] = True

    return X, y


