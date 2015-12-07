import argparse

import sys

parser = argparse.ArgumentParser(
    description="filter out all instances that are contained in the gold data provided from the TP corpus and output in the same format as the gold data")
parser.add_argument('input_file', help='input corpus files')
parser.add_argument('--gold', help='gold files', required=True, nargs='+')
parser.add_argument('--binarize', help='use only U35/O45 for age', action='store_true')

args = parser.parse_args()

gold = set()

def read_conll_file(file_name):
    """
    read in a file with format:
    word1    tag1
    ...      ...
    wordN    tagN

    Sentences MUST be separated by newlines!

    :param file_name: file to read in
    :return: generator of instances ((list of  words, list of tags) pairs)
    """
    current_words = []
    current_tags = []
    current_class = None

    for line in open(file_name):
        line = line.strip()

        if line:

            elements = line.split('\t')

            if len(elements) == 2:
                word, tag = elements
                current_words.append(word)
                current_tags.append(tag)
            else:
                current_class = line

        else:
            yield (current_class, current_words, current_tags)
            current_words = []
            current_tags = []
            current_class = None

    # if file does not end in newline (it should...), check whether there is an instance in the buffer
    if current_tags != []:
        yield (current_class, current_words, current_tags)

for file_name in args.gold:
    for (labels, words, tags) in read_conll_file(file_name):
        gold.add(' '.join(words))

removed = 0
for line in open(args.input_file):
    elements = line.strip().split("\t")

    if elements[-1] in gold:
        removed += 1
        continue

    if args.binarize:
        if elements[0] == 'NONE':
            continue
        else:
            elements[0] = int(elements[0])
            if elements[0] < 35:
                elements[0] = 'U35'
            elif elements[0] > 45:
                elements[0] = 'O45'
            else:
                continue

    print('%s\t_' % elements[-1].replace(' ', '\t_\n'))
    print('%s\n' % ','.join(elements[:-1]))
    # sys.exit()

print('removed %s items' % removed, file=sys.stderr)