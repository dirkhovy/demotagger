import argparse
import sys
import tensorflow as tf
import numpy as np
import multiprocessing
from tensorflow.python.ops import rnn
from collections import deque
from models_tf import RnnCategorial, CnnCategorial
from reader import read_datasets, make_dense


parser = argparse.ArgumentParser(
    description="train a demographics predictor. File formats: CoNLL, plus one line with CSV demographic values")
parser.add_argument('--train', help='train files', required=True)
parser.add_argument('--dev', help='dev files', required=True)
parser.add_argument('--test', help='test files', required=True)
parser.add_argument('--architecture', '-a', help="Model architecture", choices=['lstm', 'cnn'])
# parser.add_argument('--embeddings', help='embeddings files (word2vec format)', default=None)
parser.add_argument('--target', help='predict age, gender, both of them, or the joint cross-product',
                    choices=('age', 'gender', 'both', 'joint'), default='both')
# parser.add_argument('--trainer', help='which training algorithm to use',
#                     choices=('adagrad', 'sgd', 'adam', 'adadelta', 'momentum'), default='adam')
parser.add_argument('--num-epochs', help='Number of epochs', default=50)
# parser.add_argument('--status', help='number of processed instances between status updates', default=10000, type=int)
# parser.add_argument('--noise', help='amount of noise added to embeddings', default=0.1, type=float)
parser.add_argument('--dim-rnn', help='dimensionality of hidden RNN layer', default=50, type=int)
parser.add_argument('--dim-emb', help='dimensionality of word embeddings', default=100, type=int)
parser.add_argument('--dim-out', help='dimensionality of output layer', default=32, type=int)
# parser.add_argument('--dropout', help='dropout probability for final sentence representation', default=0.0, type=float)
parser.add_argument('--batch-size', help='batch size', default=1, type=int)
parser.add_argument('--max-len', help='Maximum length of input', default=100, type=int)
args = parser.parse_args()

# Read in data, mapping words to integers
datasets = read_datasets(args.train, args.test, args.dev, args.target)
train_dataset = datasets['train']
dev_dataset = datasets['dev']
test_dataset = datasets['test']
word_mapper = datasets['word_mapper']
label_mapper = datasets['label_mapper']

word_embs = tf.Variable(tf.random_uniform([len(word_mapper), args.dim_emb], -1, 1, tf.float32),
                        name='word_embs')

# Create dense representations of datasets
X_train, y_train = make_dense(train_dataset['sentences'], args.max_len,
                              train_dataset['labels'], len(label_mapper))
train_sent_lens = np.array([len(sent) for sent in train_dataset['sentences']])
X_dev, y_dev = make_dense(dev_dataset['sentences'], args.max_len,
                          dev_dataset['labels'], len(label_mapper))
dev_sent_lens = np.array([len(sent) for sent in dev_dataset['sentences']])
X_test, y_test= make_dense(test_dataset['sentences'], args.max_len,
                          test_dataset['labels'], len(label_mapper))
test_sent_lens = np.array([len(sent) for sent in test_dataset['sentences']])


assert len(X_train) == len(y_train)
assert len(X_train) == len(train_sent_lens)

# Setup model
if args.architecture == 'lstm':
    rnn_cell_type = rnn.rnn_cell.LSTMCell(args.dim_rnn, args.dim_emb)
    model = RnnCategorial(word_embs=word_embs, cell=rnn_cell_type,
                          num_labels=len(label_mapper),
                          max_seq_len=args.max_len)
else:
    model = CnnCategorial(word_embs=word_embs,
                          num_labels=len(label_mapper),
                          max_seq_len=args.max_len)


NUM_THREADS = min(10, multiprocessing.cpu_count())
with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=NUM_THREADS)) as sess:
# with tf.Session() as sess:
    # Try ADAM out for training
    global_step = tf.Variable(0, name="global_step", trainable=False)

    optimizer = tf.train.AdamOptimizer(1e-4)
    grads_and_vars = optimizer.compute_gradients(model.loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    sess.run(tf.initialize_all_variables())


    # Batches
    indices = np.arange(len(X_train))
    last_index = len(indices) - (len(indices) % args.batch_size)
    indices = indices[:last_index]

    # Validation set performance
    dev_feed = {model.y_indicator: y_dev,
                model.y: dev_dataset['labels'],
                model.input_lengths: dev_sent_lens,
                model.input: X_dev,
                model.dropout_keep_p: 1}

    # Test set performance
    test_feed = {model.y_indicator: y_test,
                model.y: test_dataset['labels'],
                model.input_lengths: test_sent_lens,
                model.input: X_test,
                model.dropout_keep_p: 1}

    stat_size = 50
    num_correct_list = deque(maxlen=stat_size)
    losses = deque(maxlen=stat_size)
    # num_total = deque(50)

    for epoch in range(1, args.num_epochs + 1):
        print("ITERATION {}".format(epoch), file=sys.stderr)
        np.random.shuffle(indices)

        # Train loop
        for i in range(0, len(indices), args.batch_size):
            batch_indices = indices[i:i+args.batch_size]

            feed_dict = {model.y_indicator: y_train[batch_indices],
                         model.y: [train_dataset['labels'][i] for i in batch_indices],
                         model.input_lengths: train_sent_lens[batch_indices],
                         model.input: X_train[batch_indices],
                         model.dropout_keep_p: 0.5}
            #
            _, loss_batch, num_correct_batch = sess.run([train_op, model.loss, model.num_correct], feed_dict)

            # num_total += len(batch_indices)
            num_correct_list.append(num_correct_batch)
            losses.append(loss_batch)
            if i % 500 == 0:
                moving_avg = sum(num_correct_list) / (len(num_correct_list) * args.batch_size)
                print("Epoch {}, instance {} (Batch size {}). Loss {} / Cumulative training accuracy {}".format(epoch, i, args.batch_size, sum(losses), moving_avg), file=sys.stderr)


        preds_dev, num_correct_dev = sess.run([model.preds, model.num_correct], dev_feed)
        preds_test, num_correct_test = sess.run([model.preds, model.num_correct], test_feed)
        print("Epoch {}: Development set performance: {}".format(epoch, num_correct_dev / len(X_dev)), file=sys.stdout, flush=True)
        print("Epoch {}: Test set performance: {}".format(epoch, num_correct_test / len(X_test)), file=sys.stdout, flush=True)

