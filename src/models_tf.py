import tensorflow as tf
from tensorflow.python.ops import rnn

def _slice_column_2d(input_, column):
    col_slice = tf.slice(input_, [0, column], [-1, 1])
    return tf.squeeze(col_slice, [1], name='col_{}_slice'.format(column))

class CategorialModel:
    def _build_pred_and_loss(self, prev_layer, num_units):
        # Dropout
        with tf.name_scope("dropout"):
            prev_layer_drop = tf.nn.dropout(prev_layer, self.dropout_keep_p)

        with tf.name_scope('output'):
            w_out = tf.Variable(tf.truncated_normal([num_units, self.num_labels], stddev=0.1),
                                name='w')
            b_out = tf.Variable(tf.constant(0.1, shape=[self.num_labels], name='b'))
            y_logits = tf.nn.xw_plus_b(prev_layer_drop, w_out, b_out, name='logits')
            losses = tf.nn.softmax_cross_entropy_with_logits(y_logits, self.y_indicator, name='softmax')
            self.loss = tf.reduce_mean(losses, name='avg_over_losses')

        with tf.name_scope('performance'):
            preds = tf.argmax(y_logits, 1, name='preds')
            self.preds = preds
            self.num_correct = tf.reduce_sum(tf.to_float(tf.equal(preds, self.y)))


class RnnCategorial(CategorialModel):
    """
    Encodes a sequence of word ids to a single output vector using an RNN.
    From the output vector, run a feed-forward network to produce a multi-class output.
    """
    def __init__(self, word_embs, cell, num_labels, max_seq_len=50):
        self.word_embs = word_embs
        self.vocab_size = word_embs.get_shape()[0].value
        self.dim_emb = word_embs.get_shape()[1].value
        self.max_seq_len = max_seq_len

        self.cell = cell
        self.num_labels = num_labels

        self._build_graph()

    def _build_graph(self):
        # Placeholders
        self.y = tf.placeholder(tf.int64)
        self.y_indicator = tf.placeholder(tf.float32, name='y_indicator')
        self.input_lengths = tf.placeholder(tf.int32, name='input_length')
        self.input = tf.placeholder(tf.int32, name='input',
                                    shape=[None, self.max_seq_len])
        self.dropout_keep_p = tf.placeholder(tf.float32)

        last_rnn_output = self._build_rnn()
        self._build_pred_and_loss(last_rnn_output, self.cell.output_size)

    def _build_rnn(self):
        word_emb_seq = []
        # Unroll for up to the maximum sequence length
        for i in range(self.max_seq_len):
            input_slice = _slice_column_2d(self.input, i)
            gather_node = tf.gather(self.word_embs, input_slice)
            word_emb_seq.append(gather_node)

        encoder_outputs, encoder_states = rnn.rnn(self.cell, word_emb_seq, sequence_length=self.input_lengths,
                                                  dtype=tf.float32)

        # Combine the outputs into a (max_seq_len, batch_size, output_size) dim. tensor.
        outputs_by_time = tf.pack(encoder_outputs)

        # Return a slice outputs_by_time[max_nonzero_index, :, :], eliminating the first dimension
        max_nonzero_index = tf.reduce_max(self.input_lengths) - 1
        begin = tf.concat(0, [tf.expand_dims(max_nonzero_index, 0), [0], [0]])
        states_by_batch = tf.squeeze(tf.slice(outputs_by_time, begin, [1, -1, -1]))

        return states_by_batch


class CnnCategorial(CategorialModel):
    def __init__(self, word_embs, num_labels, max_seq_len, num_filters=50):
        self.word_embs = word_embs
        self.vocab_size = word_embs.get_shape()[0].value
        self.dim_emb = word_embs.get_shape()[1].value
        self.max_seq_len = max_seq_len

        self.num_filters = num_filters
        self.num_labels = num_labels

        self._build_graph()

    def _build_graph(self):
        # Inputs to be supplied at train/test time
        self.y = tf.placeholder(tf.int64)
        self.y_indicator = tf.placeholder(tf.float32, name='y_indicator')
        self.input_lengths = tf.placeholder(tf.int32, name='input_length')
        self.input = tf.placeholder(tf.int32, name='input',
                                    shape=[None, self.max_seq_len])
        self.dropout_keep_p = tf.placeholder(tf.float32)

        top_conv, num_conv_units = self._build_conv_layers()
        self._build_pred_and_loss(top_conv, num_conv_units)


    def _build_conv_layers(self):
        embeds_seq = tf.nn.embedding_lookup(self.word_embs, self.input)
        # Add a 1-dim channel
        embeds_seq_with_ch = tf.expand_dims(embeds_seq, -1)

        pooled_ops = []
        num_conv_units = 0
        for filter_size in range(2, 3 + 1):
            num_conv_units += self.num_filters
            with tf.name_scope('conv-' + str(filter_size)):
                filter_shape = [filter_size, self.dim_emb, 1, self.num_filters]
                w_filt = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1))
                b_filter = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[self.num_filters]))

                conv_op = tf.nn.conv2d(embeds_seq_with_ch, w_filt,
                                       strides=[1, 1, 1, 1],
                                       padding='VALID')

                conv_act = tf.nn.relu(tf.nn.bias_add(conv_op, b_filter))

                num_applications = conv_act.get_shape()[1].value
                pooled = tf.nn.max_pool(conv_act,
                                        ksize=[1, num_applications, 1, 1],
                                        strides=[1, 1, 1, 1],
                                        padding='VALID')

                pooled_ops.append(tf.squeeze(pooled, squeeze_dims=[1, 2]))

        top_conv = tf.concat(1, pooled_ops)

        return top_conv, num_conv_units

