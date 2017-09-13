from __future__ import print_function

import time
import numpy as np
import os

import keras
from keras.datasets import mnist
from keras import backend as K

import tensorflow as tf
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import tag_constants, signature_constants


batch_size = 128
num_classes = 10

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_test = x_test.astype('float32')
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_test = keras.utils.to_categorical(y_test, num_classes)

K.clear_session()

print(y_test[0:2])

tmp_dir = os.path.join('/', 'tmp', 'mnist_saved_models')

full_model_dir = 'full_saved_model'
optimized_model_dir = 'optimized_saved_model'

version = '1'

full_model_path = os.path.join(tmp_dir, full_model_dir, version)
optimized_model_path = os.path.join(tmp_dir, optimized_model_dir, version)

PATH_TO_LOAD = full_model_path

with tf.Session(graph=tf.Graph()) as sess:
    start_time = time.time()

    populated_graph = loader.load(
        sess=sess,
        tags=[tag_constants.SERVING],
        export_dir=PATH_TO_LOAD)

    print('loading the saved_model took {}s'.format(time.time()-start_time))

    signature = populated_graph.signature_def['mnist_cnn_signature']
    input_tensor_info = signature.inputs['images']
    output_tensor_info = signature.outputs['scores']

    input_tensor_name = input_tensor_info.name
    output_tensor_name = output_tensor_info.name

    batch_delimiters = []
    batch_size = 100
    batch_instantiator = 0
    num_test_samples = x_test.shape[0]
    while num_test_samples - batch_instantiator >= batch_size:
        index_tuple = (batch_instantiator, batch_instantiator+batch_size)
        batch_delimiters.append(index_tuple)
        batch_instantiator += batch_size
    if num_test_samples - batch_instantiator > 0:
        batch_delimiters.append((batch_instantiator, num_test_samples))

    feed_dict_list = [{input_tensor_name: x_test[tup[0]:tup[1]]}
                      for tup in batch_delimiters]

    start_time = time.time()

    batch_predictions = [sess.run(output_tensor_name, feed_dict)
                         for feed_dict in feed_dict_list]

    prediction = np.concatenate(batch_predictions)
    predicted_digit = np.argmax(prediction, axis=-1)

    print('prediction took {}s'.format(time.time()-start_time))

    correct_predictions = 0
    for idx in range(num_test_samples):
        if y_test[idx][predicted_digit[idx]] == 1:
            correct_predictions += 1

    accuracy = correct_predictions / num_test_samples

    print('accuracy was {} percent over {} test samples'
          .format(accuracy, num_test_samples))
