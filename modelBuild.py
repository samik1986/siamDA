import numpy as np
import keras
import tensorflow as tf
import keras.models
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, merge, UpSampling2D, Reshape
from keras.layers import Input, TimeDistributed
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import plot_model
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
# from customLayer import *
from ncLayer import *
from keras import backend as K
from keras.engine.topology import Layer
from tensorflow.python.ops import array_ops
from scipy.linalg._expm_frechet import vec
from tensorflow.python.framework import ops
from tensorflow.python.framework.op_def_library import _Flatten, _IsListValue

# Generate dummy data
x_train = np.random.random((100, 100, 100, 3))
# x_train = np.zeros(100,100,100,3)
y_train = keras.utils.to_categorical(
    np.random.randint(10, size=(100, 1)), num_classes=10)
y_aux_train = keras.utils.to_categorical(
    np.random.randint(2, size=(100, 1)), num_classes=2)
# y_aux_train = np.zeros((100, 100, 100, 3))

x_test = np.random.random((2, 100, 100, 3))
y_test = keras.utils.to_categorical(
    np.random.randint(10, size=(2, 1)), num_classes=10)
x1_test = np.zeros((2, 100, 100, 3))
y_aux_test = keras.utils.to_categorical(
    np.random.randint(2, size=(2, 1)), num_classes=2)

# x_train = np.load('gxTrain.npy')
# y_train = np.load('gyTrain.npy')
# x_aux_train = np.load('pxTrain.npy')
# y_aux_train =np.load('vyTrain.npy')

# print y_train

input_dim =  [100,100,3]

sess = tf.InteractiveSession()

def hellinger_distance(y_true,y_pred):
    y_true = K.clip()

def create_network(input_dim):

    input_source = Input(input_dim)
    input_target = Input(input_dim)

    #---Autoencoder----

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_target)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # print encoded
    x = UpSampling2D((2, 2))(encoded)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid')(x)
    # print K.int_shape(decoded)
    # print input_source

    #----Selection Layer for Source Channel----
    merge_input_source = merge([input_source, decoded],
                          mode="concat", concat_axis=-1)
    selection_source = selectionLayer2D(input_dim)(merge_input_source)
    sel_shape = K.int_shape(selection_source)
    sel_1 = Reshape((sel_shape[1][0],sel_shape[1][1],sel_shape[1][2]),
                   input_shape=sel_shape)(selection_source)
    # print K.int_shape(sel1)

    # print merge_input_source


    # ---Input Target Channel----
    layer1_target = Conv2D(64, (3, 3),
                           activation='relu',
                           padding='same')(input_target)
    layer2_target = Conv2D(128, (3, 3),
                           activation='relu',
                           padding='same')(layer1_target)
    maxpool1_target = MaxPooling2D(pool_size=(2, 2),
                                   padding='same')(layer2_target)
    dropout1_target = Dropout(0.25)(maxpool1_target)


    #---Input Source Channel-------
    layer1_source = Conv2D(64, (3, 3),
                           activation='relu',
                           padding='same')(sel_1)
    layer2_source = Conv2D(128, (3, 3),
                           activation='relu',
                           padding='same')(layer1_source)
    maxpool1_source = MaxPooling2D(pool_size=(2, 2),
                                   padding='same')(layer2_source)
    dropout1_source = Dropout(0.25)(maxpool1_source)





    #---1st Conjugation-----
    merge_common1 = merge([dropout1_source, dropout1_target],
                         mode="concat", concat_axis=-1)


    #---Source Path---
    layer3_source = Conv2D(64, (3, 3),
                           activation='relu',
                           padding='same')(dropout1_source)

    #---Transfer 1 Source-------
    merge_source = merge([layer3_source, merge_common1],
                         mode="concat", concat_axis=-1)
    # print merge_source
    layer4_source = Conv2D(32, (3, 3),
                           activation='relu',
                           padding='same')(merge_source)
    maxpool2_source = MaxPooling2D(pool_size=(2, 2),
                                   padding='same')(layer4_source)
    layer5_source = Conv2D(32, (1, 1),
                           activation='relu',
                           padding='same')(maxpool2_source)
    dropout2_source = Dropout(0.25)(layer5_source)
    flat_source = Flatten()(dropout2_source)
    dense1_source = Dense(256,
                          activation='relu')(flat_source)



    #---Target Path---
    layer3_target = Conv2D(64, (3, 3),
                           activation='relu',
                           padding='same')(dropout1_target)

    #----Transfer 1 Target-----
    merge_target = merge([layer3_target, merge_common1],
                         mode="concat", concat_axis=-1)
    layer4_target = Conv2D(32, (3, 3),
                           activation='relu',
                           padding='same')(merge_target)
    maxpool2_target = MaxPooling2D(pool_size=(2, 2),
                                   padding='same')(layer4_target)
    layer5_target = Conv2D(32, (1, 1),
                           activation='relu',
                           padding='same')(maxpool2_target)
    dropout2_target = Dropout(0.25)(layer5_target)
    flat_target = Flatten()(dropout2_target)
    dense1_target = Dense(256, activation='relu')(flat_target)



    #---2nd Conjugation----
    merge_common2 = merge([dense1_target,dense1_source],
                         mode="concat", concat_axis=-1)
    print merge_common2
    com_distribution = MyLayer((2,256))(merge_common2)

    sel_aux = selectionLayer1D(256)(com_distribution)

    # aux_out = distribLayer(2, activation='sigmoid',
    #                 )(com_distribution)

    dropout1_common = Dropout(0.5)(sel_aux)


    dense2_common = Dense(10, activation='softmax',
                          )(dropout1_common)
    print dense2_common
    final = Model(inputs=[input_source,input_target],
                  outputs=[dense2_common])

    return final


model = create_network([100, 100, 3])
print(model.summary())
plot_model(model, to_file='models.png')
# SVG(model_to_dot(models).create(prog='dot', format='svg'))
# tbCallBack = keras.callbacks.TensorBoard(log_dir='Graph', histogram_freq=0,
#           write_graph=True, write_images=True)
#
# tbCallBack.set_model(models)
#
# keras.callbacks.TensorBoard(sess)

sgd = SGD(lr=0.001, decay=1e-6, momentum=0.7, nesterov=True)

model.compile(loss='categorical_crossentropy',
              optimizer=sgd)

model.fit([x_train, x_train],
          y_train,
          batch_size=150, epochs=5)

# score = models.evaluate([x_test, x1_test],
#                        [y_test, y_aux_test],
#                        batch_size=20)


# print score

