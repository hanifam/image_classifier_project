import warnings
warnings.filterwarnings('ignore')

import time
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_hub as hub

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

from PIL import Image
import utility_functions

import argparse

import os
import sys

# Count the arguments
arguments = len(sys.argv) - 1

my_parser = argparse.ArgumentParser(allow_abbrev=False)
# image path argument
my_parser.add_argument('Path', metavar='path', type=str, help='the path to image')
# model name argument
my_parser.add_argument('Model', metavar='model', type=str, help='the model name')
# number of classes to display
my_parser.add_argument('--top_k', action='store', type=int, help='optional argument for number of most likely classes')
# file with class name mappings
my_parser.add_argument('--category_names', action="store", dest="category_names")

# retrieve the arguments
args = my_parser.parse_args()
image_path = str(args.Path)
my_model = args.Model
K = 1 # default number of predictions displayed

    
#Load the Keras model
model = tf.keras.models.load_model(my_model, custom_objects={'KerasLayer':hub.KerasLayer})

# if the application has only two arguments then only display the most likely class and probability
# if the arguments are 4, then determine what the last two arguments are and given appropriate outputs

if arguments ==  4:
    if sys.argv[3] == '--top_k':
        K = args.top_k
        ps_topk, classes = utility_functions.predict_top_k(image_path, model, K)
        print('The top {} classes are {} with probabilities {} respectively'.format(K, classes, ps_topk))
    else:
        file_name = args.category_names
        ps, class_name = utility_functions.predict_class_name(image_path, model, file_name)
        print("The model predicted the image to be a {} with probability {}".format(class_name, ps))
elif arguments == 2:
    ps_topk, classes = utility_functions.predict_top_k(image_path, model, K)
    print('The model predicted the image to be from class {} with probability {}'.format(classes, ps_topk))
            









