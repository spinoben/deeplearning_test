# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 10:28:54 2017

@author: kb202
"""

import tensorflow as tf
import numpy as np
import time
import sys
#sys.path.append('/home/kb202/code/python/tensorflow/cifar10/')
sys.path.append('cifar10')
import cifar10, cifar10_input


max_steps = 3000
batch_size = 128
data_dir = 'cifar10/cifar10_data/cifar-10-batches-bin'

def weight(shape, stddev, w1):
    var = tf.Variable(tf.truncated_normal(shape,stddev=stddev))
    if w1 is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var), w1, name='weigth_loss')
        tf.add_to_collection('losses', weight_loss)
    return var
    
def bias(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))
    
def conv_2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1,1,1,1],padding='SAME')
    

    
cifar10.maybe_download_and_extract
images_train, labels_train = cifar10_input.distorted_inputs(data_dir=data_dir, batch_size=batch_size)
images_test, labels_test = cifar10_input.inputs(eval_data=True, data_dir=data_dir, batch_size=batch_size)
image_holder = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])
label_holder = tf.placeholder(tf.int32, [batch_size])

w_conv1 = weight([5,5,3,64], 5e-2, 0)
b_conv1 = bias([64])
h_conv1 = tf.nn.relu(conv_2d(image_holder, w_conv1)+b_conv1)
pool1 = tf.nn.max_pool(h_conv1, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')
norm1 = tf.nn.lrn(pool1, 4, bias=1,alpha=0.001/9,beta=0.75)

w_conv2 = weight([5,5,64,64], 5e-2, 0)
b_conv2 = bias([64])
h_conv2 = tf.nn.relu(conv_2d(norm1, w_conv2)+b_conv2)
pool2 = tf.nn.max_pool(h_conv2, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')
norm2 = tf.nn.lrn(pool2, 4, bias=1, alpha=0.001/9, beta=0.75)

reshape = tf.reshape(norm2, [batch_size, -1])
dim = reshape.get_shape()[1].value
w_fc1 = weight([dim, 384], 0.04, 0.004)
b_fc1 = bias([384])
h_fc1 = tf.nn.relu(tf.matmul(reshape, w_fc1)+b_fc1)

w_fc2 = weight([384, 192], 0.04, 0.004)
b_fc2 = bias([192])
h_fc2 = tf.nn.relu(tf.matmul(h_fc1, w_fc2)+b_fc2)

w_fc3 = weight([192,10], 1/192.0, 0)
b_fc3 = bias([10])
h_fc3 = tf.add(tf.matmul(h_fc2, w_fc3),b_fc3)

def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='reoss_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')
    
loss = loss(h_fc3, label_holder)
train = tf.train.AdamOptimizer(1e-4).minimize(loss)
top_k_op = tf.nn.in_top_k(h_fc3, label_holder, 1)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
tf.train.start_queue_runners()

for step in range(max_steps):
    strat_time = time.time()
    image_batch, label_batch = sess.run([images_train, labels_train])
    _, loss_value = sess.run([train, loss], feed_dict={image_holder:image_batch, label_holder:label_batch})
    duration = time.time()-strat_time
    if step%10 == 0:
        examples_per_sec = batch_size/duration
        sec_per_batch = float(duration)
        format_str = ('step %d, loss=%.2f (%.1f examples/sec; %.3f sec/batch)')
        print format_str % (step, loss_value, examples_per_sec, sec_per_batch)

num_examples = 10000
import math
num_iter = int(math.ceil(num_examples/batch_size))
true_count = 0
total_sample_count = num_iter * batch_size
step = 0
while step < num_iter:
    image_batch, label_batch = sess.run([images_test, labels_test])
    predictions = sess.run([top_k_op], feed_dict={image_holder:image_batch, label_holder:label_batch})
    true_count ==np.sum(predictions)
    step += 1
    
precision = true_count/total_sample_count
print 'precision @ 1= %.3f'%precision

