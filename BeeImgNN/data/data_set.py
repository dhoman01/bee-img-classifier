import tensorflow as tf

class data_set(object):
    """Parse and load the training images into a tensorflow dataset"""

    # `labels[i]` is the label for the image in `filenames[i]`
    def __init__(self, filenames, labels, batch_size = 25):
        dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
        dataset = dataset.map(self._parse_file)
        dataset = dataset.shuffle(buffer_size=15000)
        batched_dataset = dataset.batch(batch_size)
        batched_dataset = batched_dataset.repeat()

        self.iterator = batched_dataset.make_one_shot_iterator()
        self.next_element = self.iterator.get_next()

    def _parse_file(self, filename, label):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_png(image_string, channels=3)
        grey_scaled = tf.image.rgb_to_grayscale(image_decoded)
        return grey_scaled, label



