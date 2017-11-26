import tensorflow as tf
slim = tf.contrib.slim
trunc_normal = lambda stddev:tf.truncate_normal_initializer(0.0, stddev)

max_steps = 3000
batch_size = 128
data_dir = 'cifar10/cifar10_data/cifar-10-batches-bin'
cifar10.maybe_download_and_extract
images_train, labels_train = cifar10_input.distorted_inputs(data_dir=data_dir, batch_size=batch_size)
images_test, labels_test = cifar10_input.inputs(eval_data=True, data_dir=data_dir, batch_size=batch_size)


def inception_v3_arg_scope(weight_decay=0.0004,
                           stddev=0.1,
                           batch_norm_var_collection='moving-vars'):
    batch_norm_params={
        'decay':0.9997,
        'epsilon':0.001,
        'ipdates_collection':tf.GraphKeys.UPDATE_OPS,
        'variables_colletcions':{
            'beta':None,
            'gama':None,
            'moving_mean':[batch_norm_var_collection],
            'moving_variance':[batch_norm_var_collection]
        }
    }
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_regularizer=slim.l2_regularizer(weight_decay)):
        with slim.arg_scope([slim.conv2d],
                            weights_initializer=tf.truncate_normal_initializer(stddev=stddev),
                            activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm,
                            normalizer_param=batch_norm_params) as sc:
            return sc

def inception_v3_base(inputs, scope=None):
    end_point={}
    with tf.variable_scope(scope, 'InceptionV3',[inputs]):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.arg_poll2d], stride=1, padding='VALID'):
            net = slim.conv2d(inputs, 32, [3,3], stride=2, scope='Conv2d_1a_3x3')
            net = slim.conv2d(net, 64, [3,3], padding='SAME', scope='Conv2d_2b_3x3')
            net = slim.max_pool2d(net, [3,3], stride=2, scope='Maxpool_3a_3x3')
            net = slim.conv2d(net, 80, [1,1], scope='Conv2d_3b_1x1')
            net = slim.conv2d(net, 192, [3,3], scope='Conv2d_4a_3x3')
            net = slim.max_pool2d(net, [3,3], stride=2, scope='Maxpool_5a_3x3')

        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
            with tf.variable_scope('Mixed_5d'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 64, [1,1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 48, [1,1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 64, [5,5], scope='Conv2d_0b_5x5')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 64, [1,1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 96, [3,3], scope='Conv2d_0b_3x3')
                    branch_2 = slim.conv2d(branch_2, 96, [3,3], scope='Conv2d_0c_3x3')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3,3], scope='Avgpool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 32, [1,1], scope='Conv2d_0b_1x1')

                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

            with tf.variable_scope('Mixed_5c'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 64, [1,1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 48, [1,1], scope='Conv2d_0b_1x1')
                    branch_1 = slim.conv2d(branch_1, 64, [5,5], scope='conv_1_0c_5x5')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 64, [1,1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 96, [3,3], scope='Conv2d_0b_3x3')
                    branch_2 = slim.conv2d(branch_2, 96, [3,3], scope='Conv2d_0c_3x3')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3,3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 64, [1,1], scope='Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

            with tf.variable_scope('Mixed_5d'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 64, [1,1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 48, [1,1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 64, [5,5], scope='Conv2d_0b_3x3')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 64, [1,1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 96, [3,3], scope='Conv2d_0b_3x3')
                    branch_2 = slim.conv2d(branch_2, 96, [3,3], scope='Conv2d_0c_3x3')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3,3], scope='Avgpool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 64, [1,1], scope='Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

            with tf.variable_scope('Mixed_6a'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 384, [3,3], stride=2, padding='VALID', scope='Conv2d_1a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 64, [1,1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 96, [3,3], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 96, [3,3], stride=2, padding='VALID', scope='Conv2d_1a_1x1')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.max_pool2d(net, [3,3], stride=2, padding='VALID', scope='MaxPool_1a_3x3')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
            with tf.variable_scope('Mixed_6b'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 192, [1,1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 128, [1,1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 128, [1,7], scope='Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, 192, [7,1], scope='Conv2d_0c_7x1')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 128, [1,1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 128, [7,1], scope='Conv2d_0b_7x1')
                    branch_2 = slim.conv2d(branch_2, 128, [1,7], scope='Conv2d_0c_1x7')
                    branch_2 = slim.conv2d(branch_2, 128, [7,1], scope='Conv2d_0d_7x7')
                    branch_2 = slim.conv2d(branch_2, 192, [1,7], scope='Cpnv2d_0e_1x7')
                with tf.variable_scope('Banch_3'):
                    branch_3 = slim.avg_pool2d(net, [3,3], scope='AvgPoll_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 192, [1,1], scope='Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

            with  tf .variable_scope('Mixed_6c'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 192, [1,1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 160, [1,1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 160, [1,7], scope='Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, 192, [7,1], scope='Conv2d_0c_7x1')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 160, [1,1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 160, [7,1], scope='Conv2d_0b_7x1')
                    branch_2 = slim.conv2d(branch_2, 160, [1,7], scope='Conv2d_0c_1x7')
                    branch_2 = slim.conv2d(branch_2, 160, [7,1], scope='Conv2d_0c_7x1')
                    branch_2 = slim.conv2d(branch_2, 160, [1,7], scope='Conv2d_0d_1x7')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3,3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 192, [1,1], scope='Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

            with tf.variable_scope('Mixed_6d'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 192, [1,1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Brranch_1'):
                    branch_1 = slim.conv2d(net, 160, [1,1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 160, [1, 7], scope='Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, 160, [7, 1], scope='Conv2d_0c_7x1')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_1, 160, [7,1], scope='Conv2d_0b_7x1')
                    branch_2 = slim.conv2d(branch_1, 160, [1,7], scope='Conv2d_0c_1x7')
                    branch_2 = slim.conv2d(branch_1, 160, [7,1], scope='Conv2d_0d_7x1')
                    branch_2 = slim.conv2d(branch_1, 192, [1,7], scope='Conv2d_0e_1x7')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

            with tf.variable_scope('Mixed_6e'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 192, [1,1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Brranch_1'):
                    branch_1 = slim.conv2d(net, 192, [1,1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 192, [1, 7], scope='Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_1, 192, [7,1], scope='Conv2d_0b_7x1')
                    branch_2 = slim.conv2d(branch_1, 192, [1,7], scope='Conv2d_0c_1x7')
                    branch_2 = slim.conv2d(branch_1, 192, [7,1], scope='Conv2d_0d_7x1')
                    branch_2 = slim.conv2d(branch_1, 192, [1,7], scope='Conv2d_0e_1x7')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
            end_point['Mixed_6e'] = net;

            with tf.variable_scope('Mixed_7a'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 192, [1,1], scope='Conv2d_0a_1x1')
                    branch_0 = slim.conv2d(branch_0, 320, [3, 3], stride=2, padding='VALID', scope='Conv2d_1a_3x3')
                with tf.variable_scope('Brranch_1'):
                    branch_1 = slim.conv2d(net, 192, [1,1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 192, [1, 7], scope='Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
                    branch_1 = slim.conv2d(branch_1, 192, [3, 3], stride=2, padding='VALID', scope='Conv2d_1a_3x3')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID', scope='MaxPool_1a_3x3')
                net = tf.concat([branch_0, branch_1, branch_2], 3)

            with tf.variable_scope('Mixed_7b'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 320, [1,1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Brranch_1'):
                    branch_1 = slim.conv2d(net, 384, [1,1], scope='Conv2d_0a_1x1')
                    branch_1 = tf.concat([slim.conv2d(branch_1, 384, [1, 3], scope='Conv2d_0b_1x3'),
                                          slim.conv2d(branch_1, 384, [3, 1], scope='Conv2d_0b_3x1')], 3)
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 448, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 384, [3, 3], scope='Conv2d_0b_3x3')
                    branch_2 = tf.concat([slim.conv2d(branch_2, 384, [1, 3], scope='Conv2d_0c_1x3'),
                                          slim.conv2d(branch_2, 384, [3, 1], scope='Conv2d_0c_3x1')], 3)
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 192, [1,1], scope='Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

            with tf.variable_scope('Mixed_7c'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 320, [1,1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Brranch_1'):
                    branch_1 = slim.conv2d(net, 384, [1,1], scope='Conv2d_0a_1x1')
                    branch_1 = tf.concat([slim.conv2d(branch_1, 384, [1, 3], scope='Conv2d_0b_1x3'),
                                          slim.conv2d(branch_1, 384, [3, 1], scope='Conv2d_0b_3x1')], 3)
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 448, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 384, [3, 3], scope='Conv2d_0a_3x3')
                    branch_2 = tf.concat([slim.conv2d(branch_2, 384, [1, 3], scope='Conv2d_0c_1x3'),
                                          slim.conv2d(branch_2, 384, [3, 1], scope='Conv2d_0c_3x1')], 3)
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
            return net, end_point

def inception_v3(inputs,
                 num_classes=1000,
                 is_training=True,
                 dropout_keep_prob=0.8,
                 prediction_fn=slim.softmax,
                 spatial_spueeze=True,
                 reuse=None,
                 scope='inceptionV3'):
    with tf.variable_scope(scope, 'InceptionV3', [inputs, num_classes], reuse=reuse) as scope:
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
            net, end_point = inception_v3_base(inputs, scope=scope)
            with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
                aux_logits = end_point['Mixed_6e']
                with tf.variable_scope('AuxLogits'):
                    aux_logits = slim.avg_pool2d(aux_logits, [5,5], stride=3, padding='VALID', scope='AvgPool_1a_5x5')
                    aux_logits = slim.conv2d(aux_logits, 128, [1,1], scope='Conv2d_1b_1x1')
                    aux_logits = slim.conv2d(aux_logits, 768, [5,5], weights_initializer=trunc_normal(0.01), padding='VALID', scope='Conv2d_2a_5x5')
                    aux_logits = slim.conv2d(aux_logits, num_classes, [1,1], activation_fn=None, normalizer_fn=None, weights_initializer=trunc_normal(0.001), scope='Conv2d_2b_1x1')
                    if spatial_spueeze:
                        aux_logits = tf.squeeze(aux_logits, [1,2], name='SpatialSqueeze')
                    end_point['Auxlogits'] = aux_logits

            with tf.variable_scope('Logits'):
                net = slim.avg_pool2d(net, [8,8], padding='VALID', scope='Avgpool_1a_8x8')
                net = slim.dropout(net, keep_prob=dropout_keep_prob, scope='Dropout_1b')
                end_point['PreLogits'] = net
                logits = slim.conv2d(net, num_classes, [1,1], activation-fn=None, normalizer_fn=None, scope='Conv2d_1c_1x1')
                if spatial_spueeze:
                    logits =tf.squeeze(logits, [1,2], name='SpatialSqueeze')
            end_point['logits'] = logits
            end_point['Prediction'] = prediction_fn(logits, scope='Predictions')
    return logits, end_point

def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='reoss_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')

batch_size = 32
image = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])
label = tf.placeholder(tf.int32, [batch_size])
with slim.arg_scope(inception_v3_arg_scope()):
    logits, endpoint = inception_v3(images_train, is_training=True)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
num_batches = 100
