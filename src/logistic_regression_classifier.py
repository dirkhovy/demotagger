import argparse
from collections import Counter

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder

parser = argparse.ArgumentParser(
    description="train a demographics predictor. File formats: CoNLL, plus one line with CSV demographic values")
parser.add_argument('--train', help='train file')
parser.add_argument('--test', help='test file')
parser.add_argument('--target', help='predict age, gender, both of them, or the joint cross-product',
                    choices=('age', 'gender', 'both', 'joint'), default='both')

args = parser.parse_args()


def read_data_file(file_name):
    """
    read in a CoNLL style file
    format of files: each line is
    "word<TAB>tag<newline>", followed by
    "age,gender",
    blank line is new sentence.
    :param fname: file to read
    :return: generator of ([words, tags], [labels])
    """
    sentences = []
    sentence = []
    labels = []
    label = None

    for line in open(file_name):
        line = line.strip()

        if not line:
            if sentence:
                sentences.append(sentence)
                labels.append(label)
            sentence = []
            label = None

        else:
            elements = line.split('\t')

            # read in age and gender info
            if len(elements) == 1:
                age, gender = line.split(',')
                if args.target == 'age':
                    label = age
                elif args.target == 'gender':
                    label = gender
                # elif args.target == 'both':
                #     label = [age, gender]
                elif args.target == 'joint':
                    label = '%s-%s' % (age, gender)

            # read in words and tags
            elif len(elements) == 2:
                word, pos_tag = elements
                sentence.append(word)

            else:
                print(('Problem reading input file "%s": unexpected line "%s"' % (file_name, line)))

    return labels, sentences


def convert_instances_bow(instances, features, train=False):
    print('Converting instances...\n')

    if train:
        print('- computing word features...\n')

        word_counts = Counter()

        for line_no, instance in enumerate(instances):
            if line_no > 0:
                if line_no % 1000 == 0:
                    print("%s\n" % (line_no))
                elif line_no % 100 == 0:
                    print('.', end='')

            word_counts.update(instance)

        print('done\n')

        # top_N = min(N, len(word_counts))
        M = len(instances) * .9

        print('- keeping only features wih 1 < N < %s\n' % M)

        # mixed_feature_words = TfidfVectorizer(vocabulary=[w for (w,f) in sorted(word_counts.items(), key=lambda x:x[1], reverse=True) if f > 1 and f < M])#[:top_N])
        mixed_feature_words = TfidfVectorizer(
            vocabulary=[w for (w, f) in list(word_counts.items()) if f > 1 and f < M])  # [:top_N])
        features = mixed_feature_words

    # use an iterator to make things memory efficient
    def make_corpus(doc_files):
        for doc in doc_files:
            yield ' '.join(doc)


    print('- transforming instances...\n')
    sentences = make_corpus(instances)
    # use sparse matrix?
    out = features.fit_transform(sentences).tocsr()

    # print('- dimensionality reduction!\n')
    # pca = TruncatedSVD(n_components=500)
    # out_new = pca.fit_transform(out)
    # print('done\n\n')

    return features, out#_new


label_encoder = LabelEncoder()

# TRAIN
labels, sentences = read_data_file(args.train)
features, train_instances = convert_instances_bow(sentences, [], train=True)

# transform labels into SKLearn format
label_encoder.fit(labels)
train_labels = label_encoder.transform(labels)

classifier = LogisticRegression(class_weight="auto")
print('Training classifier...')
classifier.fit(train_instances, train_labels)

# TEST
test_labels, test_sentences = read_data_file(args.test)

_, test_instances = convert_instances_bow(test_sentences, features, train=False)

label_encoder.fit(test_labels)
test_labels_en = label_encoder.transform(test_labels)

# predicted_probs = classifier.predict_proba(test_instances)
predicted_labels = classifier.predict(test_instances)

acc = accuracy_score(test_labels_en, predicted_labels)
prec = precision_score(test_labels_en, predicted_labels)
rec = recall_score(test_labels_en, predicted_labels)
f1 = f1_score(test_labels_en, predicted_labels)
print("accuracy: %.4f,\tprec: %.4f, rec: %.4f, F1: %.4f" % (acc, prec, rec, f1))
