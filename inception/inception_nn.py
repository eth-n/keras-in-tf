import keras
from keras.applications import inception_v3
from keras import backend as K


model = inception_v3.InceptionV3(weights='imagenet', include_top=True)

# Write the model's configuration to a json file. A useful way to checkpoint
# a model after training, whether you want to
model_json = model.to_json()
with open('inception_model.json', 'w') as json_file:
    json_file.write(model_json)
model.save_weights('inception_weights.h5')
print('Model and best weights stored to disk')

K.clear_session()
