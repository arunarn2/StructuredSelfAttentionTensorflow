import tensorflow as tf
"""
Tensorflow implementation of 'A Structured Self Attentive Sentence Embedding'
(https://arxiv.org/pdf/1703.03130.pdf)
"""


class StructuredSelfAttention:
    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size,
                 hidden_size, d_a_size, r_size, fc_size, p_coef):

        # Placeholders for input, output and dropout
        with tf.name_scope("Input"):
            self.input_text = tf.placeholder(tf.int32, shape=[None, sequence_length], name='input_text')
            self.input_y = tf.placeholder(tf.float32, shape=[None, num_classes], name='input_y')

        # Length of the sequence data
        input_length = tf.cast(tf.reduce_sum(tf.sign(tf.abs(self.input_text)), reduction_indices=1), tf.int32)
        initializer = tf.contrib.layers.xavier_initializer()

        # Embeddings
        with tf.name_scope("Embedding"):
            self.w_embedding = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name="W_text")
            self.embedded_input = tf.nn.embedding_lookup(self.w_embedding, self.input_text)

        # Bidirectional(forward & backward) Recurrent Structure
        with tf.name_scope("BiLstm"):
            fw_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
            bw_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
            (self.fw_output, self.bw_output), states = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell,
                                                                                       cell_bw=bw_cell,
                                                                                       inputs=self.embedded_input,
                                                                                       sequence_length=input_length,
                                                                                       dtype=tf.float32)
            self.H = tf.concat([self.fw_output, self.bw_output], axis=2)
            h_reshape = tf.reshape(self.H, [-1, 2 * hidden_size])

        with tf.name_scope("SelfAttention"):
            # shape(W_s1) = d_a * 2u
            self.W_s1 = tf.get_variable("W_s1", shape=[2*hidden_size, d_a_size], initializer=initializer)
            # shape(W_s2) = r * d_a
            self.W_s2 = tf.get_variable("W_s2", shape=[d_a_size, r_size], initializer=initializer)
            h2 = tf.matmul(tf.nn.tanh(tf.matmul(h_reshape, self.W_s1)), self.W_s2)
            h2_reshape = tf.transpose(tf.reshape(h2, [-1, sequence_length, r_size]), [0, 2, 1])
            self.A = tf.nn.softmax(h2_reshape, name="attention")

        with tf.name_scope("SentenceEmbedding"):
            self.M = tf.matmul(self.A, self.H)

        with tf.name_scope("FullyConnected"):
            self.M_flat = tf.reshape(self.M, shape=[-1, 2 * hidden_size * r_size])
            W_fc = tf.get_variable("W_fc", shape=[2 * hidden_size * r_size, fc_size], initializer=initializer)
            b_fc = tf.Variable(tf.constant(0.1, shape=[fc_size]), name="b_fc")
            self.fc = tf.nn.relu(tf.nn.xw_plus_b(self.M_flat, W_fc, b_fc), name="fc")

        with tf.name_scope("Output"):
            W_output = tf.get_variable("W_output", shape=[fc_size, num_classes], initializer=initializer)
            b_output = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b_output")
            self.logits = tf.nn.xw_plus_b(self.fc, W_output, b_output, name="logits")
            self.predictions = tf.argmax(self.logits, 1, name="predictions")

        with tf.name_scope("Penalization"):
            self.AA_T = tf.matmul(self.A, tf.transpose(self.A, perm=[0, 2, 1]))
            self.eye = tf.reshape(tf.tile(tf.eye(r_size), [tf.shape(self.A)[0], 1]), [-1, r_size, r_size])
            # compute Frobenius norm
            self.P = tf.square(tf.norm(self.AA_T - self.eye, axis=[-2, -1], ord='fro'))

        # Calculate mean cross-entropy loss
        with tf.name_scope("Loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss_P = tf.reduce_mean(self.P * p_coef)
            self.loss = tf.reduce_mean(losses) + self.loss_P

        # Accuracy
        with tf.name_scope("Accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, axis=1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name="accuracy")
