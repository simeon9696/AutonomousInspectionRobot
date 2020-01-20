import tensorflow as tf
import numpy as np
import os
import time
import functools
#from IPython import display as ipythondisplay

import util


# Create path to ABC file
dirname = os.path.dirname(__file__)
path_to_file = os.path.join(dirname, 'data/irish.abc')



text = open(path_to_file).read()
# length of text is the number of characters in it
print('Length of text: {} characters'.format(len(text)))

util.play_generated_song(text)
