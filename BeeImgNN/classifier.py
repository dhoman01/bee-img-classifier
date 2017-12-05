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
tf.flags.DEFINE_string("input_file", "",
                       "Location of image file to classify.")

def print_img_class(img_class, sess):
  argmax = tf.argmax(img_class)
  print(img_class)
  if tf.equal(argmax, tf.argmax([1,0])).eval(session=sess):
    print('Image contains a bee')
  else:
    print('Image does not contain a bee')

sess = tf.Session('', tf.Graph())
with sess.graph.as_default():
  saver = tf.train.import_meta_graph(os.path.join(FLAGS.checkpoint_path, "model.ckpt-2000.meta"))
  saver.restore(sess, os.path.join(FLAGS.checkpoint_path, "model.ckpt-2000"))
  coord = tf.train.Coordinator()
  threads = []
  for qr in sess.graph.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
    threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))  
    
  image_string = tf.read_file(FLAGS.input_file)
  image_decoded = tf.image.decode_png(image_string, channels=3)
  img = tf.image.rgb_to_grayscale(image_decoded)
  x = tf.get_default_graph().get_tensor_by_name('Placeholder:0')
  img_class = sess.run('Variable_9:0',feed_dict={'Placeholder:0': [img.eval(session=sess)]}) 
  print_img_class(img_class, sess)
image = Image.open(FLAGS.input_file)
image.show()
