from __future__ import print_function

import time
import numpy as np
import os

import keras
from keras.datasets import cifar10
from keras import backend as K

import tensorflow as tf
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import tag_constants, signature_constants


np.random.seed(42)

# Create 250 299x299 3 channel 'images' to run the inception_v3 model on.
# These inputs are obviously random numbers, but we're just worried about
# how long the multiplications take; the math isn't concerned with what numbers
# it's working with.
input_data = np.random.rand(250, 299, 299, 3)

# Build directory structure the saved_models will be stored in
tmp_dir = os.path.join('/', 'tmp', 'inception_saved_models')

full_model_dir = 'full_saved_model'
optimized_model_dir = 'optimized_saved_model'

# Each saved_model must be versioned. Versions don't need to be integers,
# they just need to be valid directory names.
version = '1'

full_model_path = os.path.join(tmp_dir, full_model_dir, version)
optimized_model_path = os.path.join(tmp_dir, optimized_model_dir, version)

PATH_TO_LOAD = full_model_path

'''

'''

with tf.Session(graph=tf.Graph()) as sess:
    start_time = time.time()

    populated_graph = loader.load(
        sess=sess,
        tags=[tag_constants.SERVING],
        export_dir=optimized_model_path)

    print('loading the saved_model took {}s'.format(time.time()-start_time))

    signature = populated_graph.signature_def['inception_signature']
    input_tensor_info = signature.inputs['in_images']
    output_tensor_info = signature.outputs['out_images']

    input_tensor_name = input_tensor_info.name
    output_tensor_name = output_tensor_info.name

    batch_delimiters = []
    batch_size = 1
    batch_instantiator = 0
    num_test_samples = input_data.shape[0]
    while num_test_samples - batch_instantiator >= batch_size:
        index_tuple = (batch_instantiator, batch_instantiator+batch_size)
        batch_delimiters.append(index_tuple)
        batch_instantiator += batch_size
    if num_test_samples - batch_instantiator > 0:
        batch_delimiters.append((batch_instantiator, num_test_samples))

    feed_dict_list = [{input_tensor_name: input_data[tup[0]:tup[1]]}
                      for tup in batch_delimiters]

    for i in range(5):
        start_time = time.time()
        batch_predictions = [sess.run(output_tensor_name, feed_dict)
                             for feed_dict in feed_dict_list]

        prediction = np.concatenate(batch_predictions)
        print('optimized prediction took {}s'.format(time.time()-start_time))

    # predicted_digit = np.argmax(prediction, axis=-1)
    #
    # correct_predictions = 0
    # for idx in range(num_test_samples):
    #     if y_test[idx][predicted_digit[idx]] == 1:
    #         correct_predictions += 1
    #
    # accuracy = correct_predictions / num_test_samples
    #
    # print('accuracy was {} percent over {} test samples'
    #       .format(accuracy, num_test_samples))


with tf.Session(graph=tf.Graph()) as sess:
    start_time = time.time()

    populated_graph = loader.load(
        sess=sess,
        tags=[tag_constants.SERVING],
        export_dir=full_model_path)

    print('loading the saved_model took {}s'.format(time.time()-start_time))

    signature = populated_graph.signature_def['inception_signature']
    input_tensor_info = signature.inputs['in_images']
    output_tensor_info = signature.outputs['out_images']

    input_tensor_name = input_tensor_info.name
    output_tensor_name = output_tensor_info.name

    batch_delimiters = []
    batch_size = 1
    batch_instantiator = 0
    num_test_samples = input_data.shape[0]
    while num_test_samples - batch_instantiator >= batch_size:
        index_tuple = (batch_instantiator, batch_instantiator+batch_size)
        batch_delimiters.append(index_tuple)
        batch_instantiator += batch_size
    if num_test_samples - batch_instantiator > 0:
        batch_delimiters.append((batch_instantiator, num_test_samples))

    feed_dict_list = [{input_tensor_name: input_data[tup[0]:tup[1]]}
                      for tup in batch_delimiters]

    for i in range(5):
        start_time = time.time()
        batch_predictions = [sess.run(output_tensor_name, feed_dict)
                             for feed_dict in feed_dict_list]

        prediction = np.concatenate(batch_predictions)
        print('unoptimized prediction took {}s'.format(time.time()-start_time))
