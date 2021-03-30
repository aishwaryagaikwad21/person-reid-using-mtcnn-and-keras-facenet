from keras.models import load_model
model = load_model('facefeatures_new_model.h5')
print(model.inputs)
print(model.outputs)