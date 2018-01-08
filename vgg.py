import tensorflow as tf


def vgg_block(net, num_convs, channels):
    for _ in range(num_convs):
        net = tf.layers.conv2d(net, filters=channels, kernel_size=3,
                               padding='same', activation=tf.nn.relu,
                               kernel_initializer=tf.contrib.layers.xavier_initializer)
        net = tf.layers.max_pooling2d(net, 2, 2)
        return net


num_output = 10
architecture = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))



