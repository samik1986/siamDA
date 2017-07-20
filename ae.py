import numpy as np
import keras
import matplotlib.pyplot as plt
import tensorflow as tf
import keras.models
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.models import Model, load_model
from keras.layers import Dense, Dropout, Flatten, merge, UpSampling2D, Reshape, BatchNormalization
from keras.layers import Input, TimeDistributed
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import plot_model
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras import backend as K
from keras.engine.topology import Layer
from tensorflow.python.ops import array_ops
from scipy.linalg._expm_frechet import vec
from tensorflow.python.framework import ops
from tensorflow.python.framework.op_def_library import _Flatten, _IsListValue
from keras.callbacks import TensorBoard, ModelCheckpoint



# n = 500

print "loading gallery..."
x_train = np.load('/root/PycharmProjects/tifs/ae_gxTrain1.npy')
x_index = [np.random.randint(0,len(x_train)) for r in xrange(100000)]
x_train = x_train[x_index,]
x_train = x_train.astype('float32')/255.



print "loading probe..."
x_aux_train = np.load('/root/PycharmProjects/tifs/ae_pxTrain1.npy')
print "loading done..."
x_aux_train = x_aux_train[x_index,]
x_aux_train = x_aux_train.astype('float32')/255.

# fig = plt.figure()
#
# fig.add_subplot(1,2,1)
# plt.imshow(x_train[n])
# fig.add_subplot(1,2,2)
# plt.imshow(x_aux_train[n])
# plt.show()


input_dim =  [100,100,3]


def create_network(input_dim):

    # input_source = Input(input_dim)
    input_target = Input(input_dim)

    #---Autoencoder----
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(input_target)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    # x = MaxPooling2D((2, 2), padding='same')(x)
    # x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    shape1 = K.int_shape(x)
    print shape1[0]
    x = Flatten()(x)
    shape2 = K.int_shape(x)
    print shape2[0]
    x = Dense(4096,activation='relu')(x)
    x = Dense(1024,activation='relu')(x)
    x = Dense(512,activation='relu')(x)
    # encoded = Dense(256,activation='relu')(x)
    encoded = Dropout(0.5)(x)



    # print encoded
    # x = Dense(256,activation='relu')(encoded)
    x = Dense(512,activation='relu')(x)
    x = Dense(1024,activation='relu')(x)
    x = Dense(4096,activation='relu')(x)
    x = Dense(shape2[1],activation='relu')(x)
    x = Reshape([shape1[1],shape1[2],shape1[3]])(x)
    x = UpSampling2D((2, 2))(x)
    # x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    # x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu',padding='same')(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid',padding='same')(x)
    print K.int_shape(decoded)
    # print input_source


    final = Model(inputs=input_target,
                  outputs=decoded)

    return final

# with K.tf.device('/gpu:1'):
model = create_network([100, 100, 3])

print(model.summary())

model_path = 'models'

# sgd = SGD(lr=0.001, decay=1e-6, momentum=0.7, nesterov=True)
adadelta = keras.optimizers.adadelta(lr=0.001,decay=1e-5)
model.compile(loss='mse',
              optimizer='adadelta')
filepath="models/ckpt{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
# model.load_weights('models/ckpt122.hdf5')

model.fit(x_aux_train,
          x_train,
          batch_size=200, epochs=1000000,
          callbacks=[TensorBoard(log_dir='models/'),checkpoint])


