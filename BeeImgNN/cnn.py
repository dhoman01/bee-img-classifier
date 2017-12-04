import tensorflow as tf

class CNN(object):
    """A CNN to classify bee images"""
    
    def _weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def _bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def _conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def _max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def eval(self, inputs):
        #x_image = tf.reshape(inputs, [-1, 32, 32, 3])
        x_image = inputs
        W_conv1 = self._weight_variable([5, 5, 1, 32])
        b_conv1 = self._bias_variable([32])

        h_conv1 = tf.nn.relu(self._conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = self._max_pool_2x2(h_conv1)

        W_conv2 = self._weight_variable([5, 5, 32, 64])
        b_conv2 = self._bias_variable([64])

        h_conv2 = tf.nn.relu(self._conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = self._max_pool_2x2(h_conv2)

        W_conv3 = self._weight_variable([5, 5, 64, 128])
        b_conv3 = self._bias_variable([128])

        h_conv3 = tf.nn.relu(self._conv2d(h_pool2, W_conv3) + b_conv3)
        h_pool3 = self._max_pool_2x2(h_conv3)

        W_fc1 = self._weight_variable([1024 * 2, 1024])
        b_fc1 = self._bias_variable([1024])

        h_pool3_flat = tf.reshape(h_pool3, [-1, 1024 * 2])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

        self.keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        W_fc2 = self._weight_variable([1024, 2])
        b_fc2 = self._bias_variable([2])

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        return y_conv

