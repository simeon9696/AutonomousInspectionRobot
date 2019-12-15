import tensorflow as tf
# tf.enable_eager_execution() enabled by default in TF 2.0
import numpy as np
import matplotlib.pyplot as plt

is_correct_tf_version = '2.0.0' in tf.__version__
assert is_correct_tf_version, \
    "Wrong tensorflow version {} installed".format(tf.__version__)

is_eager_enabled = tf.executing_eagerly()
assert is_eager_enabled, \
    "Tensorflow eager mode is not enabled"

a = tf.constant(15, name="a")
b = tf.constant(61, name="b")

# Add them!
c = tf.add(a, b, name="c")
print(c)


# Construct a simple computation graph
def graph(a, b):
    # TODO: Define the operation for c, d, e (use tf.add, tf.subtract, tf.multiply)

    c = tf.add(a, b, name="c")
    d = tf.subtract(a, b, name="d")
    e = tf.multiply(a, b, name="e")
    return e


# Consider example values for a,b
a, b = 1.5, 2.5
# Execute the computation
e_out = graph(a, b)
print(e_out)


# n_in: number of inputs
# n_out: number of outputs
# out = sigmoid(z)
# out = sigmoid(W*x+b)
def our_dense_layer(x, n_in, n_out):
    # Define and initialize parameters, a weight matrix W and biases b
    W = tf.Variable(tf.ones((n_in, n_out)))
    b = tf.Variable(tf.zeros((1, n_out)))

    '''TODO: define the operation for z (hint: use tf.matmul)'''
    z = tf.matmul(x, W) + b

    '''TODO: define the operation forr out (hint: use tf.sigmoid)'''
    out = tf.sigmoid(z)
    return out


x_input = tf.constant([[1, 2.]], shape=(1, 2))  # What does this line do?

print(our_dense_layer(x_input, n_in=2, n_out=3))  # Call dense layer to get the ouput of the network
