from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from cnn import CNN
import os
from PIL import Image


FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string("checkpoint_path", "data/ckpt",
        "Model checkpoint file or directory containing a "
        "model checkpoint file.")
tf.flags.DEFINE_string("input_files", "",
        "Location/file patter of image file(s) to classify.")
tf.flags.DEFINE_boolean("show_img", False,
        "Show the image that is being classified")

def print_img_class(img_class, sess):
    argmax = tf.argmax(img_class, 1)
    #print(img_class)
    if tf.equal(argmax, tf.argmax([[1,0]], 1)).eval(session=sess):
        print('Image contains a bee')
    else:
        print('Image does not contain a bee')

model = CNN()

with tf.Session() as sess:
    # 1. Build and Restore Model
    model.build()
    sess.run(tf.global_variables_initializer())
    model.saver.restore(sess, tf.train.latest_checkpoint(FLAGS.checkpoint_path))

    filenames = []
    for file_pattern in FLAGS.input_files.split(","):
        filenames.extend(tf.gfile.Glob(file_pattern))
    print("Running classification on %d files matching %s" % (
                                          len(filenames), FLAGS.input_files))
    for filename in filenames: 
        # 2. Load and Pre-process Image
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_png(image_string, channels=3)
        img = tf.image.rgb_to_grayscale(image_decoded)

        # 3. Classify and Print result
        img_class = sess.run(model.logits,feed_dict={model.x: [img.eval(session=sess)], model.keep_prob: 1.0})
        print_img_class(img_class, sess)

        if FLAGS.show_img:
            image = Image.open(filename)
            image.show()
