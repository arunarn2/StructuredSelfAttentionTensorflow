import datetime
import json
import os
import numpy as np
import tensorflow as tf
import time
import re

from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

from data import batch_iterator, load_data
from model import StructuredSelfAttention

# Training parameters
tf.flags.DEFINE_integer("display_every", 100, "Number of iterations to display training info.")
tf.flags.DEFINE_integer("evaluate_every", 1000, "Evaluate model on dev set after this many steps")
tf.flags.DEFINE_integer("checkpoint_every", 1000, "Save model after this many steps")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store")
FLAGS = tf.flags.FLAGS


def json_to_dict(json_set):
    for k, v in json_set.items():
        if v == 'False':
            json_set[k] = False
        elif v == 'True':
            json_set[k] = True
        else:
            json_set[k] = v
    return json_set


with open('config.json', 'r') as f:
    params_set = json.load(f)

with open('model_params.json', 'r') as f:
    model_params = json.load(f)

params_set = json_to_dict(params_set)
model_params = json_to_dict(model_params)
print("Using parameter settings:", params_set)
print("Using model settings", model_params)
INDEX_FROM = 3


def get_coefs(word1, *arr):
    return word1, np.asarray(arr, dtype='float32')


def train():
    classification_type = params_set["classification_type"]
    init_embedding = []

    if classification_type == "multiclass":
        print("Performing multiclass classification on AGNews Dataset")
        x_text, y = load_data("data/ag_news_csv/train.csv")
        x_eval, y_eval = load_data("data/ag_news_csv/test.csv")
        x_train1, x_dev1, y_train, y_dev = train_test_split(x_text, y, test_size=0.1)

        vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(model_params['max_sentence_length'])
        x_train = np.array(list(vocab_processor.fit_transform(x_train1)))
        x_dev = np.array(list(vocab_processor.transform(x_dev1)))
        print("Text Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
        print("Train Data - X: " + str(x_train.shape) + " Labels: " + str(y_train.shape))
        print("Dev Data - X: " + str(x_dev.shape) + " Labels: " + str(y_dev.shape))

        vocab_dictionary = vocab_processor.vocabulary_._mapping
        sorted_vocab = sorted(vocab_dictionary.items(), key=lambda x: x[1])
        # w2v = word2vec.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
        # init_embedding = np.random.uniform(-1.0, 1.0, (len(vocab_processor.vocabulary_), params_set["embedding_dim"]))
        # for word, word_idx in sorted_vocab:
        #     if word in w2v:
        #         init_embedding[word_idx] = w2v[word]
        # print("Successfully loaded the pre-trained word2vec model!\n")
        del (x_train1, x_dev1)

        glove_dir = "/home/raj/Desktop/Aruna/glove.6B"
        embeddings_index = {}
        f = open(os.path.join(glove_dir, 'glove.6B.300d.txt'))
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()

        print('Total %s word vectors.' % len(embeddings_index))

        # building Hierachical Attention network
        init_embedding = np.random.random((len(vocab_processor.vocabulary_) + 1, params_set["embedding_dim"]))
        for word, i in sorted_vocab:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                init_embedding[i] = embedding_vector

        vocab_size = len(vocab_processor.vocabulary_) + 1

    elif classification_type == "binary":
        print("Performing binary classification on IMDB Dataset")
        train_set, dev_set = imdb.load_data(num_words=model_params["vocab_size"], index_from=INDEX_FROM)
        x_tr, y_tr = train_set[0], train_set[1]
        x_d, y_d = dev_set[0], dev_set[1]
        word_to_id = imdb.get_word_index()
        word_to_id = {k: (v + INDEX_FROM) for k, v in word_to_id.items()}
        word_to_id["<PAD>"] = 0
        word_to_id["<START>"] = 1
        word_to_id["<UNK>"] = 2

        # id_to_word = {value: key for key, value in word_to_id.items()}
        x_text = np.concatenate([x_tr, x_d])
        y = np.concatenate([y_tr, y_d])

        # one-hot vectors
        n_values = np.max(y) + 1
        y = np.array(np.eye(n_values)[y], int)

        n_train = x_text.shape[0] - 1000
        n_valid = 1000

        vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(model_params['max_sentence_length'])

        x_tr = x_text[:n_train]
        x_d = x_text[n_train:n_train + n_valid]

        y_train = y[:n_train]
        y_dev = y[n_train:n_train + n_valid]

        x_train = pad_sequences(x_tr, maxlen=model_params['max_sentence_length'])
        x_dev = pad_sequences(x_d, maxlen=model_params['max_sentence_length'])
        del (x_tr, y_tr, x_d, y_d, y, train_set, dev_set, x_text)

        glove_dir = "/home/raj/Desktop/Aruna/glove.6B"
        embeddings_index = {}
        f = open(os.path.join(glove_dir, 'glove.6B.300d.txt'))
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()

        print('Total %s word vectors.' % len(embeddings_index))

        # building Hierachical Attention network
        init_embedding = np.random.random((len(word_to_id) + 1, params_set["embedding_dim"]))
        for word, i in word_to_id.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                init_embedding[i] = embedding_vector
        vocab_size = len(word_to_id) + 1

    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto()
        sess = tf.Session(config=session_conf)

        with sess.as_default():
            model = StructuredSelfAttention(sequence_length=x_train.shape[1], num_classes=y_train.shape[1],
                                            vocab_size=vocab_size,
                                            embedding_size=params_set["embedding_dim"],
                                            hidden_size=model_params['lstm_hidden_dimension'],
                                            d_a_size=model_params["d_a"], r_size=params_set["attention_hops"],
                                            fc_size=model_params["fc"], p_coef=params_set["C"])

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            train_op = tf.train.AdamOptimizer(params_set["learning_rate"]).minimize(model.loss, global_step=global_step)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "outputs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", model.loss)
            acc_summary = tf.summary.scalar("accuracy", model.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Write vocabulary
            vocab_processor.save(os.path.join(out_dir, "vocabulary"))

            # Initialize all variables
            sess.run(tf.global_variables_initializer())
            sess.run(model.w_embedding.assign(init_embedding))

            # Generate batches & start training loop for each batch
            batches = batch_iterator(list(zip(x_train, y_train)), model_params["batch_size"], params_set["num_epochs"])

            for batch in batches:
                x_batch, y_batch = zip(*batch)
                feed_dict = {model.input_text: x_batch, model.input_y: y_batch}
                _, step, summaries, loss, accuracy = sess.run([train_op, global_step, train_summary_op,
                                                               model.loss, model.accuracy], feed_dict)
                train_summary_writer.add_summary(summaries, step)

                # Training progress display
                if step % FLAGS.display_every == 0:
                    time_str = datetime.datetime.now()
                    print("{}: Step {}, Loss {:g}, Acc {:g}".format(time_str, step, loss, accuracy))

                # Evaluation on validation set every 1000 steps
                if step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation on Dev set every 1000 steps:")
                    # Generate batches
                    batches_dev = batch_iterator(list(zip(x_dev, y_dev)), model_params["batch_size"], 1)
                    # Evaluation loop. For each batch...
                    loss_dev = 0
                    accuracy_dev = 0
                    cnt = 0
                    for batch_dev in batches_dev:
                        x_batch_dev, y_batch_dev = zip(*batch_dev)
                        feed_dict_dev = {model.input_text: x_batch_dev, model.input_y: y_batch_dev}

                        summaries_dev, loss, accuracy = sess.run(
                            [dev_summary_op, model.loss, model.accuracy], feed_dict_dev)
                        dev_summary_writer.add_summary(summaries_dev, step)

                        loss_dev += loss
                        accuracy_dev += accuracy
                        cnt += 1

                    time_str = datetime.datetime.now()
                    print("{}: Step {}, Loss {:g}, Acc {:g}".format(time_str, step, loss_dev / cnt, accuracy_dev / cnt))

                # Model checkpoint
                if step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=step)
                    print("Saved model checkpoint to {}\n".format(path))

        # # Get the placeholders from the graph by name
        # input_text = graph.get_operation_by_name("Input/input_text").outputs[0]
        # attn = graph.get_operation_by_name("SelfAttention/attention").outputs[0]
        #
        # # Tensors we want to evaluate
        # predictions = graph.get_operation_by_name("Output/predictions").outputs[0]
        # # Generate batches for one epoch
        # batches = batch_iterator(list(zip(x_eval, x_text)), model_params["batch_size"], 1, shuffle=False)
        # # Collect the predictions here
        # all_predictions = []
        # tokenizer = re.compile(r"[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-]+", re.UNICODE)
        # for batch in batches:
        #     x_batch, text_batch = zip(*batch)
        #
        #     batch_predictions, attention = sess.run([predictions, attn], {input_text: x_batch})
        #     all_predictions = np.concatenate([all_predictions, batch_predictions])
        #
        #     for k in range(len(attention[0])):
        #         f.write('<p style="margin:10px;">\n')
        #         ww = tokenizer.findall(text_batch[0])
        #
        #         for j in range(len(attention[0][0])):
        #             alpha = "{:.2f}".format(attention[0][k][j])
        #             if len(ww) > j:
        #                 w = ww[j]
        #             else:
        #                 break
        #
        # correct_predictions = float(sum(all_predictions == y_eval))
        # print("\nTotal number of test examples: {}".format(len(y_eval)))
        # print("Accuracy: {:g}\n".format(correct_predictions / float(len(y_eval))))


def main(_):
    train()


if __name__ == "__main__":
    tf.app.run()
