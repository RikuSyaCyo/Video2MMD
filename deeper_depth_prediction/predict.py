import argparse
import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
import cv2

import models
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def predict(model_data_path, image_path):

    
    # Default input size
    height = 228
    width = 304
    channels = 3
    batch_size = 1
   
    # Read image
    img = Image.open(image_path)
    cap = cv2.imread(image_path)
    orig_height = cap.shape[0]
    orig_width = cap.shape[1]

    background = Image.new('RGB', (width, height), (0, 0, 0))

    if(orig_width/orig_height<width/height):
        scale = height * 1.0 / orig_height
        new_width = int(orig_width * scale)
        offset = (int(round(((width - new_width)/2),0)),0)
        img = img.resize([new_width, height], Image.ANTIALIAS)
    else:
        scale = width * 1.0 / orig_width
        new_height = int(orig_height * scale)
        offset = (0,int(round(((height - new_height)/2),0)))
        img = img.resize([width, new_height], Image.ANTIALIAS)
    background.paste(img, offset)


    img = np.array(background).astype('float32')
    img = np.expand_dims(np.asarray(img), axis = 0)
   
    # Create a placeholder for the input image
    input_node = tf.placeholder(tf.float32, shape=(None, height, width, channels))

    # Construct the network
    net = models.ResNet50UpProj({'data': input_node}, batch_size, 1, False)
        
    with tf.Session() as sess:

        # Load the converted parameters
        print('Loading the model')

        # Use to load from ckpt file
        saver = tf.train.Saver()     
        saver.restore(sess, model_data_path)

        # Use to load from npy file
        #net.load(model_data_path, sess) 

        # Evalute the network for the given image
        pred = sess.run(net.get_output(), feed_dict={input_node: img})
        
        # Plot result
        fig = plt.figure()
        ii = plt.imshow(pred[0,:,:,0], interpolation='nearest')
        fig.colorbar(ii)
        plt.show()
        
        return pred
        
                
def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='Converted parameters for the model')
    parser.add_argument('image_paths', help='Directory of images to predict')
    args = parser.parse_args()

    # Predict the image
    pred = predict(args.model_path, args.image_paths)
    
    os._exit(0)

if __name__ == '__main__':
    main()

        



