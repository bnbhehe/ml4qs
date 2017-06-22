
# coding: utf-8

# In[176]:

from __future__ import division
from __future__ import print_function
import pandas as pd
# get_ipython().magic(u'pylab inline')
import pylab
import time
import warnings
import os
from collections import Counter
from itertools import chain
import json
import seaborn as sns
import ast
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)

print("Imports done ...✈✈✈✈✈✈✈✈ You are free to go.")
import keras
from keras.layers import Activation,Dense
from keras.models import Sequential
import tensorflow as tf
import six
import numpy as np
import math

# In[177]:

dataset_folder = './crowdsignals'
fname = 'chapter5_result.csv'
dataset_path = os.path.join(dataset_folder,fname)
print(dataset_path)
print(os.getcwd())
print(os.listdir(dataset_folder))


# #### Chapter 6.2
# Explore gradient descent and change the learning rate η. What happens to the trajectory of the gradient descent? Is a fast gradient descent always a good idea? (Hint: the answer is No!).
# 
# 
# ------
# In most classification problems, the cost function we want to approximate is highly convex. So to approximate it fast we might need a large learning rate. However this can backfire as the error loss gets bouncing around the downhill slope toward the local minima. An example is illustrated in the picture below. Note that these images are an ideal case for illustrating how the learning rate affects the optimization of our cost function.
# 
# <img src="sgd.png">
# In addition we show a coding example on the crowdsignal data with a simple MLP implemented in keras to show how sgd learning rate affects training.

# In[178]:

def mlp_model(D,nb_classes=10):
    
    model = Sequential()
    input_shape = (D,)
    print(input_shape,' input dimensions')
    layers = [
        Dense(100,name='fc1',input_shape=input_shape),
        Activation('tanh',name='fc1_act'),
        Dense(20,name='fc2'),
        Activation('tanh',name='fc2_act'),
        Dense(nb_classes,name='fc_out'),
        Activation('softmax',name='output')
    ]
    for layer in layers:
        print('Added layer:%s'%layer.name)
        model.add(layer)

    return model

def batch_indices(batch_nb, data_length, batch_size):
    """
    This helper function computes a batch start and end index
    :param batch_nb: the batch number
    :param data_length: the total length of the data being parsed by batches
    :param batch_size: the number of inputs in each batch
    :return: pair of (start, end) indices
    """
    # Batch start and end index
    start = int(batch_nb * batch_size)
    end = int((batch_nb + 1) * batch_size)

    # When there are not enough inputs left, we reuse some to complete the
    # batch
    if end > data_length:
        shift = end - data_length
        start -= shift
        end -= shift

    return start, end

def model_train(sess, x, y, predictions, X_train, Y_train,args=None,verbose=True):

    
    # Define loss
    loss = model_loss(y, predictions)
    
    train_step = tf.train.GradientDescentOptimizer(learning_rate=args['learning_rate']).minimize(loss)

    with sess.as_default():

        for epoch in six.moves.xrange(args['nb_epochs']):

            nb_batches = int(np.ceil(float(len(X_train)) / args['batch_size']))

            prev = time.time()
            for batch in range(nb_batches):
                start, end = batch_indices(
                    batch, len(X_train), args['batch_size'])

                train_step.run(feed_dict={x: X_train[start:end],
                                          y: Y_train[start:end]})
            assert end >= len(X_train)  # Check that all examples were used
            cur = time.time()
            if verbose:
                print("\t====================>Epoch took " + str(cur - prev) + " seconds")
            prev = cur
    return True
                

def model_loss(y, model, mean=True):

    op = model.op
    if "softmax" in str(op).lower():
        logits, = op.inputs
    else:
        logits = model

    out = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)

    if mean:
        out = tf.reduce_mean(out)
    return out
def data_prepare(df_name):
    
    df = pd.read_csv(df_name,index_col=0)
    x_columns = ['acc_phone_x','acc_phone_y','acc_phone_z',
                 'acc_watch_x','acc_watch_y','acc_watch_z',
                 'gyr_phone_x','gyr_phone_y','gyr_phone_z',
                 'hr_watch_rate',
                ]
    y_columns = ['labelOnTable','labelSitting','labelWashingHands',
                 'labelWalking','labelStanding','labelDriving',
                 'labelEating','labelRunning'
                ]
    X = df[x_columns]
    Y = df[y_columns]
    return X,Y

    
   


# In[187]:

X,Y = data_prepare(dataset_path)

keras.layers.core.K.set_learning_phase(0)
keras.backend.manual_variable_initialization(True)
# Set TF random seed to improve reproducibility
tf.set_random_seed(42)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
keras.backend.set_session(sess)
classes = np.arange(Y.shape[1])
D = (X.shape[1])
x = tf.placeholder(tf.float32, shape=(None, D))
y = tf.placeholder(tf.float32, shape=(None, len(classes)))
print("Created TensorFlow session and set Keras backend.")
train_params = { 
        'nb_epochs': 1000,
        'batch_size': 128,
        'learning_rate': 1e-3
    }

print('Number of classes %d'%len(classes))

model = mlp_model(D,nb_classes=len(classes))
preds = model(x)
model_train(sess,x,y,preds,X,Y,args=train_params)

        


# In[ ]:




# In[100]:




# In[ ]:




# In[ ]:



