import random
import argparse

parser = argparse.ArgumentParser(
    description="split the data into train, dev, test, maybe shuffle first")
parser.add_argument('data', help='train files')
parser.add_argument('--shuffle', help='shuffle data before split', action="store_true")
parser.add_argument('--devtest', help='size of dev and test', type=int)
args = parser.parse_args()

data = list(open(args.data).readlines())

if args.shuffle:
    random.shuffle(data)


dev = data[:args.devtest]
test = data[args.devtest: args.devtest*2]
train = data[args.devtest*2:]

dev_file = open('%s.dev' % args.data, 'w')
dev_file.write(''.join(dev))
dev_file.close()

test_file = open('%s.test' % args.data, 'w')
test_file.write(''.join(test))
test_file.close()

train_file = open('%s.train' % args.data, 'w')
train_file.write(''.join(train))
train_file.close()
