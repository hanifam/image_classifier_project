#Create the process_image function
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
    
def process_image(image):
    image_size = 224
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255
    image.numpy()
    return image

# Create the predict function for top_k
def predict_top_k(image_path, model, top_k):    
    
    im = Image.open(image_path)
    test_image = np.asarray(im)
    image = process_image(test_image)
    
    # this is the image that will be printed
    old_image = image
    
    # add extra dimension so that image can work with the mode    
    image = np.expand_dims(image, axis=0)  
    
    # predict the image
    ps = model.predict(image)
    
    # get the indices for sorted images
    b = np.argsort(ps)      
    c = b[0][::-1]    
    c = c[0:top_k]  # top classes
    
    # get the top K prediction probabilities
    ps_topk = ps[0][c]
    class_number = c
    
      
    return ps_topk, class_number     

# Create the predict function for class_names
def predict_class_name(image_path, model, file_name):    
    
    im = Image.open(image_path)
    test_image = np.asarray(im)
    image = process_image(test_image)
    
    # load class names 
    with open(file_name, 'r') as f:
        class_names = json.load(f)
    
    # this is the image that will be printed
    old_image = image
    
    # add extra dimension so that image can work with the mode    
    image = np.expand_dims(image, axis=0)  
    
    # predict the image
    ps = model.predict(image)
    
    # get the indices for sorted images
    b = np.argsort(ps)      
    c = b[0][::-1]     
    c = c[0]  # only keep the top prediction
    
    # get the prediction probability
    ps_topk = ps[0][c]
    
    # get the labels for the top K probabilites in class_names -- increment the value in c to get the right key   
    class_names_topk = class_names[str(c+1)]
    #for val in c:
        #class_names_topk.append(class_names[str(val+1)])    
       
    return ps_topk, class_names_topk