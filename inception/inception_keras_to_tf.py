from __future__ import print_function

import os
import shutil
import time

import keras
from keras.models import model_from_json
from keras import backend as K

import tensorflow as tf
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import utils
from tensorflow.python.saved_model import tag_constants, signature_constants


json_model = os.path.relpath('./inception_model.json')
inception_weights = os.path.relpath('./inception_weights.h5')

json_file = open(json_model, 'r')
model_json = json_file.read()

K.set_learning_phase(1)

start_time = time.time()
model = model_from_json(model_json)
model.load_weights(inception_weights)
print('time to load `learning_phase`=1 inception v3: `{}`s'.format(
    time.time() - start_time))

keras1_graph_def = K.get_session().graph_def
keras1_node_names = [node.name for node in keras1_graph_def.node]
num_keras1_nodes = len(keras1_node_names)
print('number of nodes in graph when `learning_phase`=1:', num_keras1_nodes)

K.clear_session()
K.set_learning_phase(0)

start_time = time.time()
model = model_from_json(model_json)
model.load_weights(inception_weights)
print('time to load `learning_phase`=0 inception v3: `{}`s'.format(
    time.time() - start_time))

keras0_graph_def = K.get_session().graph_def
keras0_node_names = [node.name for node in keras0_graph_def.node]
num_keras0_nodes = len(keras0_node_names)
print('number of nodes in graph when `learning_phase`=0:', num_keras0_nodes)

print('input Tensor:', model.input)
print('output Tensor:', model.output)

'''
Directory structure of a saved_model

saved_model_name/
    version_name/
        protobuf_name.pb
        variables/
            variables.data-x-of-y
            ...
            variables.index
    version_name_2/
        ...
    ...
    version_n/
        ...
'''

tmp_dir = os.path.join('/', 'tmp', 'inception_saved_models')

# Name saved_model to be in tmp_dir: saved_model_name = full_saved_model
full_model_dir = 'full_saved_model'
# Name saved_model to be in tmp_dir: saved_model_name = optimized_saved_model
optimized_model_dir = 'optimized_saved_model'

version = '1'

full_model_path = os.path.join(tmp_dir, full_model_dir, version)
optimized_model_path = os.path.join(tmp_dir, optimized_model_dir, version)

# saved_model_builder won't overwrite a version, so these will clear out
# the directories if they already exist to save space in these examples
if os.path.isdir(full_model_path):
    shutil.rmtree(full_model_path)
    print('full model version `{}` replaced'.format(version))
if os.path.isdir(optimized_model_path):
    shutil.rmtree(optimized_model_path)
    print('optimized model version `{}` replaced'.format(version))

# Use the Keras model metadata to access input and output information
input_tensor_name = model.input.name.split(':')[0]
output_tensor_name = model.output.name.split(':')[0]
print('input tensor name:', input_tensor_name)
print('output tensor name:', output_tensor_name)


input_tensor_info = utils.build_tensor_info(model.input)
output_tensor_info = utils.build_tensor_info(model.output)

signature = signature_def_utils.build_signature_def(
    inputs={'in_images': input_tensor_info},
    outputs={'out_images': output_tensor_info})


builder_1 = saved_model_builder.SavedModelBuilder(full_model_path)

with K.get_session() as sess:
    builder_1.add_meta_graph_and_variables(
        sess=sess,
        tags=[tag_constants.SERVING],
        signature_def_map={'inception_signature': signature})

    builder_1.save(as_text=False)
    print('full model saved to:', full_model_path)

    inception_node_names = [node.name for node in sess.graph_def.node]
    num_saved_nodes = len(inception_node_names)
    print('number of nodes in full_saved_model', num_saved_nodes)

K.clear_session()


with tf.Session(graph=tf.Graph()) as sess:
    start_time = time.time()
    loader.load(
        sess=sess,
        tags=[tag_constants.SERVING],
        export_dir=full_model_path)
    print('time to load unoptimized graph def: `{}`s'.format(
        time.time() - start_time))

    optimized_graph_def = tf.graph_util.convert_variables_to_constants(
        sess=sess,
        input_graph_def=sess.graph_def,
        output_node_names=[output_tensor_name])


builder_2 = saved_model_builder.SavedModelBuilder(optimized_model_path)

with tf.Session(graph=tf.Graph()) as sess:
    start_time = time.time()
    tf.import_graph_def(optimized_graph_def, name='')
    print('time to load optimized graph def: `{}`s'.format(
        time.time() - start_time))
    builder_2.add_meta_graph_and_variables(
        sess=sess,
        tags=[tag_constants.SERVING],
        signature_def_map={'inception_signature': signature})

    builder_2.save(as_text=False)
    print('Model with variables as constants saved to:', optimized_model_path)

    constant_node_names = [node.name for node in sess.graph_def.node]
    # print(constant_node_names)
    num_saved_constant_nodes = len(constant_node_names)
    print('number of nodes in optimized_saved_model', num_saved_constant_nodes)
