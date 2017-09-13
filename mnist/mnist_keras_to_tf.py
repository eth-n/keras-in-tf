from __future__ import print_function

import os
import shutil

import keras
from keras.models import model_from_json
from keras import backend as K

import tensorflow as tf
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import utils
from tensorflow.python.saved_model import tag_constants, signature_constants

from tensorflow.tools.graph_transforms import TransformGraph


json_model = os.path.relpath('./mnist_model.json')
best_weights = os.path.relpath('./mnist_best_weights.h5')

K.set_learning_phase(1)

json_file = open(json_model, 'r')
model_json = json_file.read()
model = model_from_json(model_json)
model.load_weights(best_weights)


keras1_graph_def = K.get_session().graph_def
keras1_node_names = [node.name for node in keras1_graph_def.node]
num_keras1_nodes = len(keras1_node_names)
print('number of nodes in graph when `learning_phase`=1:', num_keras1_nodes)

K.clear_session()
K.set_learning_phase(0)

model = keras.models.model_from_json(model_json)
model.load_weights(best_weights)

keras0_graph_def = K.get_session().graph_def
keras0_node_names = [node.name for node in keras0_graph_def.node]
num_keras0_nodes = len(keras0_node_names)
print('number of nodes in graph when `learning_phase`=0:', num_keras0_nodes)

print('input Tensor:', model.input)
print('output Tensor:', model.output)

tmp_dir = os.path.join('/', 'tmp', 'mnist_saved_models')

full_model_dir = 'full_saved_model'
optimized_model_dir = 'optimized_saved_model'

version = '1'

full_model_path = os.path.join(tmp_dir, full_model_dir, version)
optimized_model_path = os.path.join(tmp_dir, optimized_model_dir, version)

if os.path.isdir(full_model_path):
    shutil.rmtree(full_model_path)
    print('full model version `{}` replaced'.format(version))
if os.path.isdir(optimized_model_path):
    shutil.rmtree(optimized_model_path)
    print('optimized model version `{}` replaced'.format(version))

input_tensor_name = model.input.name.split(':')[0]
output_tensor_name = model.output.name.split(':')[0]

print('input tensor name:', input_tensor_name)
print('output tensor name:', output_tensor_name)

input_tensor_info = utils.build_tensor_info(model.input)
output_tensor_info = utils.build_tensor_info(model.output)

signature = signature_def_utils.build_signature_def(
    inputs={'images': input_tensor_info},
    outputs={'scores': output_tensor_info})

builder_1 = saved_model_builder.SavedModelBuilder(full_model_path)

with K.get_session() as sess:
    builder_1.add_meta_graph_and_variables(
        sess=sess,
        tags=[tag_constants.SERVING],
        signature_def_map={'mnist_cnn_signature': signature})

    builder_1.save(as_text=False)
    print('full model saved to:', full_model_path)

    inference_node_names = [node.name for node in sess.graph_def.node]
    num_saved_nodes = len(inference_node_names)
    print('number of nodes in the saved_model', num_saved_nodes)

K.clear_session()


with tf.Session(graph=tf.Graph()) as sess:
    loader.load(
        sess=sess,
        tags=[tag_constants.SERVING],
        export_dir=full_model_path)

    constant_graph_def = tf.graph_util.convert_variables_to_constants(
        sess=sess,
        input_graph_def=sess.graph_def,
        output_node_names=[output_tensor_name])

    input_graph_def = constant_graph_def
    input_names = [input_tensor_name]
    output_names = [output_tensor_name]
    transforms = ['strip_unused_nodes',
                  'remove_nodes(op=Identity, op=CheckNumerics)',
                  'fold_constants(ignore_errors=True']

    optimized_graph_def = TransformGraph(input_graph_def,
                                         input_names,
                                         output_names,
                                         transforms)

builder_2 = saved_model_builder.SavedModelBuilder(optimized_model_path)

with tf.Session(graph=tf.Graph()) as sess:
    # tf.import_graph_def(constant_graph_def, name='')
    tf.import_graph_def(optimized_graph_def, name='')

    builder_2.add_meta_graph_and_variables(
        sess=sess,
        tags=[tag_constants.SERVING],
        signature_def_map={'mnist_cnn_signature': signature})

    builder_2.save(as_text=False)
    print('Model with variables as constants saved to:', optimized_model_path)

    constant_node_names = [node.name for node in sess.graph_def.node]
    print(constant_node_names)
    num_saved_constant_nodes = len(constant_node_names)
    print('number of nodes in the saved_model', num_saved_constant_nodes)
