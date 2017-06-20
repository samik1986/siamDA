from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
import numpy as np
import keras
import theano
import theano.tensor as T
import tensorflow as tf
import keras.models
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, merge
from keras.layers import Input, TimeDistributed
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import plot_model
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
# from customLayer import MyLayer
from keras import backend as K
from keras.engine.topology import Layer
from tensorflow.python.ops import array_ops
from scipy.linalg._expm_frechet import vec
from tensorflow.python.framework import ops
from tensorflow.python.framework.op_def_library import _Flatten, _IsListValue



class MyLayer(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(MyLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def my_func(self,x,n):
        # for i in range(len(x)):
        with tf.Session as sess:
            p = sess.run(x)

        print p
        c = tf.cos(tf.multiply
                    (tf.subtract
                     (tf.multiply(2.0,tf.transpose
                     (tf.linspace(1.0,n,n))),1),
                    tf.divide(np.pi, tf.multiply(2.0,n)
                    )))
        v = tf.Variable([])
        for i in range(0, 255,23):
            print i
            v = tf.stack([v,x[i]],-1)
        T = tf.concat([tf.zeros([n,1]),tf.ones([n,1])],-1)
        y = tf.divide(tf.reduce_sum(x),n)
        for j in range(1,1):
            y = tf.stack([y,tf.constant(0.)],-1)
        a = tf.constant(1.)
        print(T[:,1],"....Yes")
        l = []
        # x_m = tf.expand_dims(x, 1)
        for k in range(2,n):
            T = tf.concat([T[:,1],
                              tf.subtract(
                                  tf.multiply(tf.multiply(a,c),T[:,1]),T[:,0])
                                  ],-1)
            print("yes...")
            y[k] = tf.divide(tf.multiply(tf.add_n(tf.multiply(T[:,1],x)),2),n)
            a = tf.Constant(2.)
        return l



    def call(self, inputs):
        # e1 = x[0]
        print ("123...123")


        def mu_nf(x):
            return x ** 8
        # p=tf.placeholder("int32",shape=[1])
        # p1 = tf.assign([-1],p)
        # with tf.Session as sess:
        #     sess.run(p1,feed_dict={})
        input_shape = K.int_shape(inputs)
        ip = T.matrix('ip')
        N = T.iscalar('N')
        # print(ip)
        outputs, updates = theano.scan(
            fn = mu_nf,
            sequences=ip,
            n_steps=ip.shape[0])
        mu_op = theano.function(
            inputs=[ip],
            outputs=outputs
        )
        print "hi"

        o_var = mu_op(inputs)

        print(type(o_var))

        # print(y)
        # p = tf.placeholder("float32", shape=input_shape)
        # print(p)
        # for i in xrange
        # iden = tf.Variable(tf.convert_to_tensor(np.eye(input_shape), dtype=tf.float32))
        # p1 = tf.matmul(p,iden)
        # with tf.Session() as sess:
        #     r = sess.run(p1,feed_dict={p:x})
        # r = tf.reshape(x,[-1,2,256])
        # #print(r)
        # for i in xrange(input_shape[0]):
        #     p1 = r[-1,0,:]
        #     p2 = r[-1,1,:]
        # np1 = tf.subtract(
        #     tf.divide(
        #         tf.subtract(
        #             p1,
        #             tf.reduce_min(p1)
        #         ),
        #         tf.subtract(
        #             tf.reduce_max(p1),
        #             tf.reduce_min(p1)
        #         )
        #     ),
        # 0.5)
        #
        # np2 = tf.subtract(
        #     tf.divide(
        #         tf.subtract(
        #             p2,
        #             tf.reduce_min(p2)
        #         ),
        #         tf.subtract(
        #             tf.reduce_max(p2),
        #             tf.reduce_min(p2)
        #         )
        #     ),
        # 0.5)
        # c1 = self.my_func(np1, 11)
        #inp = tf.linspace()
        # inp = np1
        # with tf.Session as sess:
        #     sess.run(init)
        #     p = sess.run(y)
        # c1 = tf.py_func(self.my_func,[inp],[tf.float32])
        # c2 = tf.py_func(self.my_func,[p2],[tf.float32])

        # for i in xrange(input_shape[0]):
        #     p = x[i, :, :entity_location[0, i, 0], :]
        # shapedX = K.reshape(x,(-1,input_shape[1:]))
        # entity_location = K.permute_dimensions(x,(1, 0))
        # y, _ = theano.scan(lambda _x, _el: _x[:, _el[0, 0]], sequences=(x, entity_location))
        # r =
        # print(r,"....YES....")
        return x


        # input_shape = K.int_shape(inputs[0])
        # timesteps = input_shape[2]
        # x = K.reshape(x, (-1,input_shape[1]*input_shape[0]) + input_shape[2:])
        # new_input_shape = K.int_shape(x)
        # r = K.eval(x[1])
        # sess = tf.Session()

        # r = np.s
        # for i in range(new_input_shape[2]):  # cant do this currently
        #      p1 = x[:, :, i]
        # entity_location = K.permute_dimensions(x, (1, 0, 2))
        # input_list = array_ops.unstack(inputs)
        # d = tf.Variable(tf.random_normal([64,625]), name='d')
        # #e2 = x[1]
        # batch_size = K.shape(e1)[0]
        # k = self.output_dim
        # e2 = K.transpose(e1)
        # e3 = K.variable(e2, dtype="float32")
        # e2 = K.batch_flatten(x)
        # e4 = K.eval(x)
        # r = K.flatten(e2)
        # s = K.shape(e2)[0]
        # for i in range(0,k-1):
        #      d.assign(K.batch_flatten(e2[i]))
        #p = K.batch_flatten(e2)
        # print(r)
        # s = np.shape(x)
        # im_vec = []
        # sess = tf. Session()
        # init = tf.global_variables_initializer()
        # K.set_session(sess)
        # K.eval(init)
        # # r=K.reshape(x,[s[1]*s[2],s[3]])
        # print(r)
        #r = ops._get_graph_from_inputs(_Flatten(x))
        # with session:
        #     r = tf.eval(x)
        # for i in xrange(len(x)[0]):
        #     for j in xrange(0,s[3]):
        #         im_vec = x[0][i][j]

        # input_seq = Input(x.get_shape())
        # flat1 = Flatten()(input_seq)
        # print(flat1)
        # flat_layer = Model(input=input_seq, outputs=flat1)
        # TD = keras.layers.TimeDistributed(flat_layer)(input_seq)
        # print(TD)
        # return x



    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

class selectionLayer1D(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(selectionLayer1D, self).__init__(**kwargs)

    def build(self, input_shape):
        super(selectionLayer1D, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        P = x[:,1,:]
        G = x[:,0,:]
        # print K.sum(G)
        if K.sum(G)==tf.constant(0.):
            op = G
            print "gallery"
        else:
            op = P
            print "probe"
        return op

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

class selectionLayer2D(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(selectionLayer2D, self).__init__(**kwargs)

    def build(self, input_shape):
        super(selectionLayer2D, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        # print x
        input_shape = K.int_shape(x)
        # print input_shape[1]
        r_x = tf.reshape(x,[-1,tf.to_int32(input_shape[1]),
                            tf.to_int32(input_shape[2]),
                            tf.to_int32(input_shape[3]/2),2])
        input_gallery = r_x[:,:,:,:,0]
        trans_gallery = r_x[:,:,:,:,1]
        # print input_gallery, trans_gallery
        # input_gallery = tf.zeros(input_shape)
        if K.sum(input_gallery) == tf.constant(0.):
            op = input_gallery[:,:,:,:]
            # print "Gallery"
        else:
            op = trans_gallery[:,:,:,:]
            # print "T-gallery"
        # print "----"
        # print K.int_shape(op)
        return op

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

class distribLayer(Layer):

    def __init__(self, output_dim, activation=None, **kwargs):
        self.output_dim = output_dim
        self.activation = activations.get(activation)
        super(distribLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(distribLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def diffMetric(self, P,G):
        return tf.norm(tf.subtract(P,G))

    def call(self, x):
        sess1 = tf.InteractiveSession()
        P = x[:,1,:]
        # print P
        G = x[:,0,:]
        # print G
        L = self.diffMetric(P,G)
        op = self.activation(L)
        # print op
        return op


    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)