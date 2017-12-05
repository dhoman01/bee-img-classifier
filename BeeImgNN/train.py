import tensorflow as tf
from data.data_set import data_set
from cnn import CNN
import os

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string("train_dir", "data/train",
                       "Directory containing the training data.")
tf.flags.DEFINE_string("test_dir", "data/test",
                       "Directory containing the test data.")
tf.flags.DEFINE_string("checkpoint_path", "data/ckpt",
                       "File, file pattern, or directory of checkpoints")
tf.flags.DEFINE_integer("epochs", 80000,
                       "Number of training iterations")
tf.flags.DEFINE_integer("save_step", 1000,
                       "Number of steps in between checkpoints")

def add_filenames_and_labels(fn_arr, lbl_arr, label, dir):
    for file in os.listdir(dir):
        if file.endswith(".png"):
            fn_arr.append(os.path.join(dir, file))
            lbl_arr.append(label)

train_filenames = []
train_labels = []
add_filenames_and_labels(train_filenames, train_labels, [1,0], FLAGS.train_dir + "/single_bee_train")
add_filenames_and_labels(train_filenames, train_labels, [0,1], FLAGS.train_dir + "/no_bee_train")

test_filenames = []
test_labels = []
add_filenames_and_labels(test_filenames, test_labels, [1,0], FLAGS.test_dir + "/single_bee_test")
add_filenames_and_labels(test_filenames, test_labels, [0,1], FLAGS.test_dir + "/no_bee_test")

model = CNN()
train_data = data_set(tf.constant(train_filenames), tf.constant(train_labels))
test_data = data_set(tf.constant(test_filenames), tf.constant(test_labels))

x = tf.placeholder(tf.float32, shape=[None, 32, 32, 1])
y_ = tf.placeholder(tf.float32, shape=[None, 2])

y_conv = model.eval(x)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Add ops to save and restore all the variables.
saver = tf.train.Saver()
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  #saver.restore(sess, FLAGS.checkpoint_path + "/model.ckpt")
  for i in range(FLAGS.epochs):
    train_img, train_lbl = sess.run(train_data.next_element)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: train_img, y_: train_lbl, model.keep_prob: 1.0})
        print('step %d, training accuracy %g' % (i, train_accuracy))
    if i % FLAGS.save_step == 0:
        saver.save(sess, FLAGS.checkpoint_path + "/model.ckpt", i)
    train_step.run(feed_dict={x: train_img, y_: train_lbl, model.keep_prob: 0.5})
  total_acc = 0;
  for it in range(16000):
    test_images, test_labels = sess.run(test_data.next_element)
    acc = accuracy.eval(feed_dict={x: test_images, y_: test_labels, model.keep_prob: 1.0})
    if it % 100 == 0:
        print('test accuracy %g' % acc)
    total_acc += acc
  print('Avg. test accuracy %g' % (total_acc / 16000))
  save_path = saver.save(sess, FLAGS.checkpoint_path + "/model.ckpt", FLAGS.epochs)
  print('Model saved at ' + save_path)
