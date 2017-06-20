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
from ncLayer import *
from keras import backend as K
from keras.engine.topology import Layer
from tensorflow.python.ops import array_ops
from scipy.linalg._expm_frechet import vec
from tensorflow.python.framework import ops
from tensorflow.python.framework.op_def_library import _Flatten, _IsListValue
from keras.callbacks import TensorBoard, ModelCheckpoint


# Generate dummy data
# x_train = np.random.random((100, 100, 100, 3))
# # x_train = np.zeros(100,100,100,3)
# y_train = keras.utils.to_categorical(
#     np.random.randint(10, size=(100, 1)), num_classes=10)
# y_aux_train = keras.utils.to_categorical(
#     np.random.randint(2, size=(100, 1)), num_classes=2)
# # y_aux_train = np.zeros((100, 100, 100, 3))
#
# x_test = np.random.random((2, 100, 100, 3))
# y_test = keras.utils.to_categorical(
#     np.random.randint(10, size=(2, 1)), num_classes=10)
# x1_test = np.zeros((2, 100, 100, 3))
# y_aux_test = keras.utils.to_categorical(
#     np.random.randint(2, size=(2, 1)), num_classes=2)

x_train = np.load('ae_gxTrain.npy')
# x_train = x_train[1:200]
# y_train = np.load('gyTrain.npy')
x_aux_train = np.load('ae_pxTrain.npy')
# x_aux_train = x_aux_train[1:200]
# y_aux_train =np.load('vyTrain.npy')
x_test = np.load('testImagesIITM.npy')
# print y_train

x_train = x_train.astype('float32')
x_train = (x_train-x_train.min())/(x_train.max()-x_train.min())
x_aux_train = x_aux_train.astype('float32')
x_aux_train = (x_aux_train-x_aux_train.min())/(x_aux_train.max()-x_aux_train.min())
# x_aux_train = x_aux_train * 255
# x_aux_train = x_aux_train.astype('uint8')
x_test = x_test.astype('float32')
x_test = (x_test-x_test.min())/(x_test.max()-x_test.min())


# x_train = x_train.astype('float32') / 255.
# x_aux_train = x_aux_train.astype('float32') / 255.
# x_test = x_test.astype('float32') / 255.

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

# x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
# x_aux_train = x_aux_train.reshape((len(x_aux_train), np.prod(x_aux_train.shape[1:])))

input_dim =  [100,100,3]

sess = tf.InteractiveSession()

def hellinger_distance(y_true,y_pred):
    y_true = K.clip()

def create_network(input_dim):

    # input_source = Input(input_dim)
    input_target = Input(input_dim)

    #---Autoencoder----
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_target)
    x = Conv2D(48, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(96, (3, 3), activation='relu', padding='same')(x)
    # x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    # x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    shape1 = K.int_shape(x)
    print shape1[0]
    x = Flatten()(x)
    shape2 = K.int_shape(x)
    print shape2[0]
    # x = Dropout(0.5)(x)
    x = Dense(4096,activation='relu')(x)
    x = Dense(1024,activation='relu')(x)
    x = Dense(512,activation='relu')(x)
    encoded = Dense(256,activation='relu')(x)
    # encoded = Dropout(0.5)(x)



    # print encoded
    x = Dense(256,activation='relu')(encoded)
    x = Dense(512,activation='relu')(x)
    x = Dense(1024,activation='relu')(x)
    x = Dense(4096,activation='relu')(x)
    x = Dense(shape2[1],activation='relu')(x)
    x = Reshape([shape1[1],shape1[2],shape1[3]])(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)
    # x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    # x = Conv2D(128, (3, 3), activation='relu',padding='same')(x)
    x = Conv2D(96, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(48, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu')(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid')(x)
    print K.int_shape(decoded)
    # print input_source


    final = Model(inputs=input_target,
                  outputs=decoded)

    return final


model = create_network([100, 100, 3])
model.save("model.h5",overwrite=True)
print(model.summary())
# plot_model(model, to_file='model.png')
# SVG(model_to_dot(model).create(prog='dot', format='svg'))
# tbCallBack = keras.callbacks.TensorBoard(log_dir='Graph', histogram_freq=0,
#           write_graph=True, write_images=True)
#
# tbCallBack.set_model(model)
#
# keras.callbacks.TensorBoard(sess)

# sgd = SGD(lr=0.00001, decay=1e-6, momentum=0.9, nesterov=True)
adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-05)
model.compile(loss='mean_squared_error',
              optimizer=adam)

datagen.fit(x_aux_train)
filepath="models/ckpt{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
model.fit_generator(datagen.flow(x_aux_train, x_train, batch_size=100),
                    steps_per_epoch=len(x_aux_train) / 200, epochs=10000,
                    callbacks=[TensorBoard(log_dir='/tmp/autoencoder',
                                 write_images=True, write_grads=True),checkpoint])

# model.fit(x_aux_train,
#           x_train,
#           batch_size=200, epochs=1000000,
#           callbacks=[TensorBoard(log_dir='/tmp/autoencoder',
#                                  histogram_freq=1,
#                                  write_images=True, write_grads=True)
#                      ])

# grads = model.optimizer.get_gradients(model.total_loss,model.get_weights())
# print sess.run(grads)
# decoded_imgs = model.predict(x_test)
# np.save('dec_images',decoded_imgs)
# n = 5
# plt.figure(figsize=(20, 4))
# for i in range(n):
#     # display original
#     ax = plt.subplot(2, n, i)
#     plt.imshow(x_test[i].reshape(100, 100))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
#
#     # display reconstruction
#     ax = plt.subplot(2, n, i + n)
#     plt.imshow(decoded_imgs[i].reshape(100, 100))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
# plt.show()

model.save("model.h5",overwrite=True)
# model.save_weights("model_weights.h5", overwrite=True)
# score = model.evaluate(x_test,
#                        y_test,
#                         batch_size=20)


# print score

