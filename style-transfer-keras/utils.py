from keras.applications import inception_v3
from keras import backend as K
model = inception_v3.InceptionV3(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)

for layer in model.layers:
    print (layer.name)
