from keras.layers import Input, Dense
from keras.models import Model
from newLayer import *
import numpy as np

# This returns a tensor
input_dim =  [100,100,3]
input_source = Input(input_dim)

cLayer = MyLayer(256)(input_source)

final = Model(inputs=input_source,
                  outputs=cLayer)

print final.summary()