import tensorflow as tf
import model as md


# Definitionen für selbst generierte Daten (Modelle, Graphen, ...)
models_path = '/home/spl/workspace/object_recognition/spl_detector/data/models/'
'''
# BHuman Model
model_name = 'encoder.hdf5'
model = tf.keras.models.load_model(models_path + model_name)

print(model.summary())

json_string = model.to_json()
with open(models_path + model_name + '.json', 'w') as f:
  f.write(json_string)
'''
# My Model
#model = md.TestModel()
model = md.SingleNaoModel((60, 80), False)
model.save(models_path + 'test_model.hdf5', save_format='h5')
print(model.summary())

json_string = model.to_json()
with open(models_path + 'test_model' + '.json', 'w') as f:
  f.write(json_string)
'''
# Single Nao 4 Model
model = tf.keras.models.load_model(models_path + 'single_nao_4_best.hdf5')

print(model.summary())

json_string = model.to_json()
with open(models_path + 'single.hdf5' + '.json', 'w') as f:
  f.write(json_string)
'''