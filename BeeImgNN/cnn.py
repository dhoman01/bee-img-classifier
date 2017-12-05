import tensorflow as tf

class CNN(object):
    """A CNN to classify bee images"""
    def __init__(self):
        self.x = tf.placeholder(tf.float32, shape=[None, 32, 32, 1], name='input_img')
        self.y_ = tf.placeholder(tf.float32, shape=[None, 2], name='img_class')

    def _weight_variable(self, shape, name=''):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name=name)

    def _bias_variable(self, shape, name=''):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name)

    def _conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def _max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def build(self):
        with tf.variable_scope("model", initializer=tf.random_uniform_initializer()) as scope:
            W_conv1 = self._weight_variable([5, 5, 1, 32], 'input_weights')
            b_conv1 = self._bias_variable([32], 'input_bias')

            h_conv1 = tf.nn.relu(self._conv2d(self.x, W_conv1) + b_conv1)
            h_pool1 = self._max_pool_2x2(h_conv1)

            W_conv2 = self._weight_variable([5, 5, 32, 64], 'hidden_1_weights')
            b_conv2 = self._bias_variable([64], 'hidden_1_bias')

            h_conv2 = tf.nn.relu(self._conv2d(h_pool1, W_conv2) + b_conv2)
            h_pool2 = self._max_pool_2x2(h_conv2)

            W_conv3 = self._weight_variable([5, 5, 64, 128], 'hidden_2_weights')
            b_conv3 = self._bias_variable([128], 'hidden_2_bias')

            h_conv3 = tf.nn.relu(self._conv2d(h_pool2, W_conv3) + b_conv3)
            h_pool3 = self._max_pool_2x2(h_conv3)

            W_fc1 = self._weight_variable([1024 * 2, 1024], 'fc_1_weights')
            b_fc1 = self._bias_variable([1024], 'fc_1_bias')

            h_pool3_flat = tf.reshape(h_pool3, [-1, 1024 * 2])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

            W_fc2 = self._weight_variable([1024, 2], 'fc_2_weights')
            b_fc2 = self._bias_variable([2], 'fc_2_bias')

            self.logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        with tf.variable_scope("training", initializer=tf.random_uniform_initializer()) as scope:
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.logits))
            self.train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

            self.correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.y_, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        self.saver = tf.train.Saver()

