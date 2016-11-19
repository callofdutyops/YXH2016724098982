"""Routine for decoding the eye files."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

# Process images of this size. Note that this differs from the original eye
# image size of 512 x 512. This smaller size can reduce consuming of computer memory.
# If one alters this number, then the entire model architecture will change and
# any model would need to be retrained.
IMAGE_SIZE = 32

# Global constants describing the eye data set.
NUM_CLASSES = 7
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 315
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 35

# The folder names of eye images.
LABLES = ['AMD', 'BRVO_CRVO', 'CSC', 'DR', 'ENormal', 'FOthers', 'GPM']


def read_eye(filename_queue):
    """Reads and parses examples from eye files.

    Args:
      filename_queue: A queue of strings with the filenames to read from.

    Returns:
      An object representing a single example, with the following fields:
        height: number of rows in the result (512)
        width: number of columns in the result (512)
        depth: number of color channels in the result (3)
        key: a scalar string Tensor describing the filename & record number
          for this example.
        label: an int32 Tensor with the label in the range 0..6.
        uint8image: a [height, width, depth] uint8 Tensor with the image data
    """

    class EYERecord(object):
        pass

    result = EYERecord()

    # Dimensions of the images in the eye dataset.
    label_bytes = 1
    result.height = 512
    result.width = 512
    result.depth = 3
    # image_bytes = result.height * result.width * result.depth

    # Read a record, getting filenames from the filename_queue.
    reader = tf.WholeFileReader()
    result.key, value = reader.read(filename_queue)

    # The first bytes represent the label, which we convert from uint8->int32.
    # LABLES.index(str(result.key.eval(session=tf.get_default_session())).split('_')[0])
    # Convert from a string to a vector of uint8 that is record_bytes long.
    label_decoded = tf.decode_raw(result.key, tf.uint8)

    # The first bytes represent the label, which we convert from uint8->int32.
    # Note that the number 55 is the length of str
    # "/home/hp/Documents/DeepLearning/MyProjects/Data/eye/[tr,te]/". We
    # only want to get the folder name after this str so we do a slice on it.
    # And also, the ops of "add(result.label, -65)" is to get the label in right range
    # which is 0..6.
    result.label = tf.cast(tf.slice(label_decoded, [61], [label_bytes]), tf.int32)
    result.label = tf.add(result.label, -65)

    result.uint8image = tf.image.decode_png(value)

    return result


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
    """Construct a queued batch of images and labels.

    Args:
      image: 3-D Tensor of [height, width, 3] of type.float32.
      label: 1-D Tensor of type.int32
      min_queue_examples: int32, minimum number of samples to retain
        in the queue that provides of batches of examples.
      batch_size: Number of images per batch.
      shuffle: boolean indicating whether to use a shuffling queue.

    Returns:
      images: Images. 4D tensor of [batch_size, height, width, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.
    """
    # Create a queue that shuffles the examples, and then
    # read 'batch_size' images + labels from the example queue.
    num_preprocess_threads = 4
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        images, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)

    # Display the training images in the visualizer.
    tf.image_summary('images', images)

    return images, tf.reshape(label_batch, [batch_size])


def distorted_inputs(data_dir, batch_size):
    """Construct distorted input for eye training using the Reader ops.

    Args:
      data_dir: Path to the eye data directory.
      batch_size: Number of images per batch.

    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.
    """
    data_dir = os.path.join(data_dir, 'tr')
    filenames = []
    for folder_name in LABLES:
        folder_path = os.path.join(data_dir, folder_name)
        filenames += [os.path.join(folder_path, f) for f in os.listdir(folder_path)]

    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer(filenames)

    # Read examples from files in the filename queue.
    read_input = read_eye(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    # Image processing for training the network. Note the many random
    # distortions applied to the image.

    # Randomly crop a [height, width] section of the image.
    distorted_image = tf.random_crop(reshaped_image, [height, width, 3])

    # Randomly flip the image horizontally.
    distorted_image = tf.image.random_flip_left_right(distorted_image)

    # Because these operations are not commutative, consider randomizing
    # the order their operation.
    distorted_image = tf.image.random_brightness(distorted_image,
                                                 max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image,
                                               lower=0.2, upper=1.8)

    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_whitening(distorted_image)

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.1
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                             min_fraction_of_examples_in_queue)
    print('Filling queue with %d eye images before starting to train. '
          'This will take a few minutes.' % min_queue_examples)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(float_image, read_input.label,
                                           min_queue_examples, batch_size,
                                           shuffle=True)


def inputs(eval_data, data_dir, batch_size):
    """Construct input for eye evaluation using the Reader ops.

    Args:
      eval_data: bool, indicating if one should use the train or eval data set.
      data_dir: Path to the eye data directory.
      batch_size: Number of images per batch.

    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.
    """
    if not eval_data:
        data_dir = os.path.join(data_dir, 'tr')
        filenames = []
        for folder_name in LABLES:
            folder_path = os.path.join(data_dir, folder_name)
            filenames += [os.path.join(folder_path, f) for f in os.listdir(folder_path)]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    else:
        data_dir = os.path.join(data_dir, 'te')
        filenames = []
        for folder_name in LABLES:
            folder_path = os.path.join(data_dir, folder_name)
            filenames += [os.path.join(folder_path, f) for f in os.listdir(folder_path)]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer(filenames)

    # Read examples from files in the filename queue.
    read_input = read_eye(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    # Image processing for evaluation.
    # Crop the central [height, width] of the image.
    resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                           width, height)

    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_whitening(resized_image)
    # Fix the shape of Tensor
    float_image.set_shape([height, width, 3])

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.1
    min_queue_examples = int(num_examples_per_epoch *
                             min_fraction_of_examples_in_queue)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(float_image, read_input.label,
                                           min_queue_examples, batch_size,
                                           shuffle=False)
