"""
Now, instead of explicitly defining a simple function,
we'll use the Keras API to define our neural network.
This will be especially important as we move on to more
complicated network architectures.
"""

# Import relevant packages
import tensorflow as tf
# tf.enable_eager_execution() enabled by default in TF 2.0
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

is_correct_tf_version = '2.0.0' in tf.__version__
assert is_correct_tf_version, \
    "Wrong tensorflow version {} installed".format(tf.__version__)

is_eager_enabled = tf.executing_eagerly()
assert is_eager_enabled, \
    "Tensorflow eager mode is not enabled"
# Define the number of inputs and outputs
n_input_nodes = 2
n_output_nodes = 3

# First define the model
model = Sequential()

# Remember: dense layers are defined by the parameters W and b!
dense_layer = Dense(n_output_nodes, input_shape=(n_input_nodes,), activation='sigmoid')

# Add dense layer to model
model.add(dense_layer)

# Test model with example input
x_input = tf.constant([1, 2.], shape=(1, 2))
print(model(x_input))

# Automatic Differentiation
# Necessary for training model with backpropagation
# All forward pass operations get recorded to a tape
# The gradient is computed by playing the tape backwards and then discarded
# A particular tf.GradientTape cam only compute one gradient
# The example below uses AutoDiff and stocastic gradient descent (SGD)
# to find the minimum of y=(x-1)^2. Gradient descent is used to optimize neural networks

x = tf.Variable([tf.random.normal([1])])
# x = tf.Variable([tf.random.normal([1])])
print("Initalizing x={}".format(x.numpy()))
learning_rate = 1e-2
history = []

for i in range(500):
    with tf.GradientTape() as tape:
        y = (x - 1)**2  # record the forward pass on the tape

    grad = tape.gradient(y, x)  # compute the gradient of y wrt x
    new_x = learning_rate * grad  # sgd update
    x.assign_sub(new_x)  # update the value of x
    history.append(x.numpy()[0])

plt.plot(history)
plt.plot([0, 500], [1, 1])
plt.legend(('Predicted', 'True'))
plt.xlabel('Iteration')
plt.ylabel('x value')
plt.show()
