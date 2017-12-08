import tensorflow as tf
from data.data_set import data_set
from cnn import CNN
import os
from progress.bar import Bar

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string("train_dir", "data/train",
        "Directory containing the training data")
tf.flags.DEFINE_string("test_dir", "data/test",
        "Directory containing the test data")
tf.flags.DEFINE_string("checkpoint_path", "data/ckpt",
        "File, file pattern, or directory of checkpoints")
tf.flags.DEFINE_integer("epochs", 80000,
        "Number of training iterations")
tf.flags.DEFINE_integer("save_step", 1000,
        "Number of steps in between checkpoints")
tf.flags.DEFINE_integer("test_steps", 16000,
        "Number of steps to run the test and build accuracy")
tf.flags.DEFINE_boolean("train", True,
        "Flag for training the model")
tf.flags.DEFINE_boolean("test", True,
        "Flag for testing the model")

def add_filenames_and_labels(fn_arr, lbl_arr, label, dir):
    for file in os.listdir(dir):
        if file.endswith(".png"):
            fn_arr.append(os.path.join(dir, file))
            lbl_arr.append(label)

train_filenames = []
train_labels = []
add_filenames_and_labels(train_filenames, train_labels, [1,0], os.path.join(FLAGS.train_dir, "single_bee_train"))
add_filenames_and_labels(train_filenames, train_labels, [0,1], os.path.join(FLAGS.train_dir, "no_bee_train"))

test_filenames = []
test_labels = []
add_filenames_and_labels(test_filenames, test_labels, [1,0], os.path.join(FLAGS.test_dir, "single_bee_test"))
add_filenames_and_labels(test_filenames, test_labels, [0,1], os.path.join(FLAGS.test_dir, "no_bee_test"))

model = CNN()
train_data = data_set(tf.constant(train_filenames), tf.constant(train_labels))
test_data = data_set(tf.constant(test_filenames), tf.constant(test_labels))

def train_model(sess):
    """Train the model on the given training data for the given number of epochs in FLAGS"""
    bar = Bar("Training", max=FLAGS.epochs, suffix='%(percent)d%% - %(eta)ds')
    for i in range(FLAGS.epochs):
        train_img, train_lbl = sess.run(train_data.next_element)
        if i % FLAGS.save_step == 0:
            train_accuracy = model.accuracy.eval(feed_dict={model.x: train_img, model.y_: train_lbl, model.keep_prob: 1.0})
            print('\nstep %d, training accuracy %g' % (i, train_accuracy))
            model.saver.save(sess, os.path.join(FLAGS.checkpoint_path, "model.ckpt"), i)
        model.train_step.run(feed_dict={model.x: train_img, model.y_: train_lbl, model.keep_prob: 0.5})
        bar.next()
    bar.finish()
    save_path = model.saver.save(sess, os.path.join(FLAGS.checkpoint_path, "model.ckpt"), FLAGS.epochs)
    print('Model saved at ' + save_path)

def test_model(sess):
    bar = Bar("Testing", max=FLAGS.test_steps, suffix='%(percent)d%% - %(eta)ds')
    total_acc = 0;
    for i in range(FLAGS.test_steps):
        test_images, test_labels = sess.run(test_data.next_element)
        acc = model.accuracy.eval(feed_dict={model.x: test_images, model.y_: test_labels, model.keep_prob: 1.0})
        if i % 1000 == 0:
            print('\ntest accuracy %g' % acc)
        total_acc += acc
        bar.next()
    bar.finish()
    print('Avg. test accuracy %g' % (total_acc / FLAGS.test_steps))

# Add ops to save and restore all the variables.
with tf.Session() as sess:
    model.build()
    sess.run(tf.global_variables_initializer())

    print('Trying to restore model')
    try:
        model.saver.restore(sess, tf.train.latest_checkpoint(FLAGS.checkpoint_path))
        print('Model restored')
    except:
        print('Could not restore model from checkpoint ' + FLAGS.checkpoint_path)

    if FLAGS.train:
        train_model(sess)
    if FLAGS.test:
        test_model(sess)
