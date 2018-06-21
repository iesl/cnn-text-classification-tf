#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn

# Parameters
# ==================================================

# Data loading params
##########tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")

tf.flags.DEFINE_string("dataset_type", "ST", "either QT (question type) or ST (sentiment)")
tf.flags.DEFINE_string("test_data_file", "./data/Question_type_split/testing_wo_stop", "Data source for the positive data.")
tf.flags.DEFINE_string("positive_test_data_file", "./data/rt-polaritydata_split/rt-polarity.pos_test_wo_stop", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_test_data_file", "./data/rt-polaritydata_split/rt-polarity.neg_test_wo_stop", "Data source for the negative data.")
tf.flags.DEFINE_string("training_data_file","./data/rt-polaritydata_split/rt-polarity.random_train_wo_stop","Data source of training data")
tf.flags.DEFINE_integer("training_data_size",2000, "number of samples in training dataset")
tf.flags.DEFINE_boolean("random_shuffle",True,"whether random shuffle training data")

# Model Hyperparameters
tf.flags.DEFINE_string("word2vec", None, "Word2vec file with pre-trained embeddings (default: None)")
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 1000, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 2, "Number of checkpoints to store (default: 5)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# Setup for shuffle data and store outputs
method = "random"
if not FLAGS.random_shuffle:
    method = os.path.basename(FLAGS.training_data_file)
time_str = datetime.datetime.now().isoformat()
output_file_name = "./log/"+FLAGS.dataset_type+'_'+method+'_num_sample_'+str(FLAGS.training_data_size)+'_'+time_str


# Data Preperation
# ==================================================

# Load data
print("Loading data...")
####x_text, y = data_helpers.load_data_and_labels(FLAGS.positive_test_data_file, FLAGS.negative_test_data_file) ############# Added parameters

if FLAGS.dataset_type == "ST":
    x_train_raw, y_train = data_helpers.load_sorted_data_and_labels(FLAGS.training_data_file)
    x_test_raw, y_dev = data_helpers.load_data_and_labels(FLAGS.positive_test_data_file, FLAGS.negative_test_data_file)
elif FLAGS.dataset_type == "QT":
    x_train_raw, y_train = data_helpers.load_QT_data_and_labels(FLAGS.training_data_file)
    x_test_raw, y_dev = data_helpers.load_QT_data_and_labels(FLAGS.test_data_file)
x_text = x_train_raw + x_test_raw

# Build vocabulary
max_document_length = max([len(x.split(" ")) for x in x_text])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array(list(vocab_processor.fit_transform(x_text)))
x_train = x[:len(x_train_raw)]
x_dev = x[len(x_train_raw):]

# Randomly shuffle data
print FLAGS.random_shuffle
np.random.seed(int(time.time()))
if FLAGS.random_shuffle == True:
    shuffle_indices = np.random.permutation(np.arange(len(y_train)))
    x_train = x_train[shuffle_indices]
    y_train = y_train[shuffle_indices]

x_train = x_train[:FLAGS.training_data_size]
y_train = y_train[:FLAGS.training_data_size]

# Split train/test set
# TODO: This is very crude, should use cross-validation
#dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
#x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
#y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

del x #, y, x_shuffled, y_shuffled

print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    


# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],   
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
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
        vocab_processor.save(os.path.join(out_dir, "vocab"))

        # Initialize all variables
        sess.run(tf.global_variables_initializer())
        if FLAGS.word2vec:
            # initial matrix with random uniform
            initW = np.random.uniform(-0.25,0.25,(len(vocab_processor.vocabulary_), FLAGS.embedding_dim))

            # load any vectors from the word2vec
            print("Load word2vec file {}\n".format(FLAGS.word2vec))
            with open(FLAGS.word2vec, "rb") as f:
                header = f.readline()
                vocab_size, vector_size = map(int, header.split())
                binary_len = np.dtype('float32').itemsize * vector_size
                #print "binary_len", binary_len
                for line in xrange(vocab_size):
                    word = []
                    while True:
                        ch = f.read(1)
                        #print "ch", ch
                        if ch == ' ':
                            word = ''.join(word)
                            break
                        if ch != '\n':
                            word.append(ch)      
                    #print "\n\n\n WORD", word
                    idx = vocab_processor.vocabulary_.get(word)
                    #print "idx", idx
                    if idx != None:	######## Changed 0 to None
                        initW[idx] = np.fromstring(f.read(binary_len), dtype='float32')  
                    else:
                        f.read(binary_len)    
            sess.run(cnn.W.assign(initW))

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                    }
            _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
            time_str = datetime.datetime.now().isoformat()
            current_epoch = float(FLAGS.batch_size)*step/FLAGS.training_data_size
            if step % 10 == 0:
                print("{}: epoch {}, step {}, loss {:g}, acc {:g}".format(time_str,current_epoch, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: 1.0
                }
            step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: random {}, step {}, loss {:g}, acc {:g}".format(time_str,FLAGS.random_shuffle, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)
            return accuracy

        # Generate batches
        batches = data_helpers.batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)

        # Training loop. For each batch,
        best_dev_acc = 0
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                dev_acc = dev_step(x_dev, y_dev, writer=dev_summary_writer)
                best_dev_acc = max(dev_acc,best_dev_acc)
                print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))

        dev_acc=dev_step(x_dev, y_dev, writer=dev_summary_writer)
        best_dev_acc = max(dev_acc,best_dev_acc)
        print best_dev_acc
                
delim = '\t'
with open(output_file_name,'w+') as f_out:
	f_out.write(FLAGS.dataset_type+delim+method+delim+str(FLAGS.training_data_size)+delim+str(best_dev_acc)+'\n')

'''
def main(argv=None):
    x_train, y_train, vocab_processor, x_dev, y_dev = preprocess()
    train(x_train, y_train, vocab_processor, x_dev, y_dev)

if __name__ == '__main__':
    tf.app.run()
'''
