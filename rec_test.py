# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import numpy as np
import sys
import cv2
import argparse
import sys
caffe_root = '/opt/caffe-apollo/'  # this file should be run from {caffe_root}/examples (otherwise change this line)
sys.path.insert(0, caffe_root + 'python')
import os
import caffe
import math
from os import walk
from os.path import join
# If you get "No module named _caffe", either you have not built pycaffe or you have the wrong path.

CLASSES = ("Black", "Red", "Yellow", "Green")



def det(image,transformer,net):
    
    transformed_image = transformer.preprocess('data', image)
    #plt.imshow(image)

    net.blobs['data'].data[...] = transformed_image

    ### perform classification
    output = net.forward()

    res = output['prob'][0]  # the output probability vector for the first image in the batch
    
    #print(res.shape)
    return res

def is_imag(filename):
    return filename[-4:] in ['.png', '.jpg']

def main(args):    
 
    caffe.set_mode_cpu()
    model_def = args.model_def
    model_weights = args.model_weights
    
    net = caffe.Net(model_def,      # defines the structure of the model
                    model_weights,  # contains the trained weights
                    caffe.TEST)     # use test mode (e.g., don't perform dropout)


    mu = np.array([69.06, 66.58, 66.56])   #  rgb
    #mu = np.array([66.56, 66.58, 69.06])
    # create transformer for the input called 'data'
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

    transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
  
    transformer.set_raw_scale('data', 2.55)      # rescale from [0, 1] to [0, 255]
    transformer.set_mean('data', mu*0.01)            # subtract the dataset-mean value in each channel
    #transformer.set_raw_scale('data', 0.01)      # rescale from [0, 1] to [0, 255]
    transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR
    #transformer.set_transpose('data', (1,2,0))  # move image channels to outermost dimension
    

    net.blobs['data'].reshape(1,        # batch size
                              3	,         # 3-channel (BGR) images
                              #args.image_resize, args.image_resize)  # image size is 227x227
                              96, 32)
    
    filenames = os.listdir(args.image_dir)
    images = filter(is_imag, filenames)
    for image in images :
        pic = args.image_dir + image
        input = caffe.io.load_image(pic)  
      #  print('python load image shape: ', input)
        #input = cv2.imread(pic)
        image_show =cv2.imread(pic)  
        result = det(input,transformer,net)
        ind = np.argmax(result)
        print(image, CLASSES[ind], result[ind])
        #vis_detections(image_show,result,result2)
        
        
def parse_args():
    parser = argparse.ArgumentParser()
    '''parse args'''
    parser.add_argument('--image_dir', default='./rec_imgs/')
    parser.add_argument('--model_def', default='./recognize/deploy.prototxt')
    parser.add_argument('--model_weights', default='./recognize/baidu_iter_250000.caffemodel')

    return parser.parse_args()
    
if __name__ == '__main__':
    main(parse_args())
