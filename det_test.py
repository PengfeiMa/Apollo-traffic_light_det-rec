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

def vis_detections(image,result) :
    h_scale = image.shape[1]/256.0
    w_scale = image.shape[0]/256.0  
    
    left = float(result[1]) * h_scale
    top = float(result[2]) * w_scale
    right = float(result[3]) * h_scale
    bot = float(result[4]) * w_scale
    label = str(result[5:].argmax())
    print(h_scale, w_scale,left, top, right, bot)

    cv2.rectangle(image,(int(left), int(top)),(int(right),int(bot)),(0,255,0), 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    size = cv2.getTextSize(label, font, 0.5, 0)[0]
    cv2.rectangle(image,(int(left), int(top)),
    (int(left+size[0]),int(top+ size[1])),(0,255,0), -1)
    cv2.putText(image, label,(int(left+0.5), int(top+ size[1]+0.5)),font,0.5,(0,0,0),0)

    return image

   

def det(image,transformer,net):
    
    transformed_image = transformer.preprocess('data', image)
    #plt.imshow(image)
   # print("image.shape: ",image.shape)
   # im_info = np.array([image.shape[1], image.shape[0], 1, 1, 0, 0]).reshape(1,6,1,1)
    im_info = np.array([image.shape[1], image.shape[0], 1, 1, 0, 0]).reshape(1,6,1,1)
    net.blobs['data'].data[...] = transformed_image
    net.blobs['im_info'].data[...] = im_info
    ### perform classification
    output = net.forward()

    res = output['bboxes'][0]  # the output probability vector for the first image in the batch
    
    #print(res.shape)
    return res

def is_imag(filename):
    return filename[-4:] in ['.png', '.jpg']

def main(args):    
 
    caffe.set_mode_gpu()
    caffe.set_device(0)
    model_def = args.model_def
    model_weights = args.model_weights
    
    net = caffe.Net(model_def,      # defines the structure of the model
                    model_weights,  # contains the trained weights
                    caffe.TEST)     # use test mode (e.g., don't perform dropout)

    mu = np.array([122.7717, 115.9465, 102.9801])   #  rgb
    #mu = np.array([66.56, 66.58, 69.06])
    # create transformer for the input called 'data'
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    #print("data blob shape: ", net.blobs['img'].data.shape)
    #print("im_info blob shape: ", net.blobs['im_info'].data.shape)
    transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
  
    transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
    transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
   # transformer.set_raw_scale('data', 0.1)      # rescale from [0, 1] to [0, 255]
    transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR
    
    

    net.blobs['data'].reshape(1,        # batch size
                              3	,         # 3-channel (BGR) images
                              #args.image_resize, args.image_resize)  # image size is 227x227
                              256, 256)
    
    filenames = os.listdir(args.image_dir)
    images = filter(is_imag, filenames)
    for image in images :
        #print(image)
        pic = args.image_dir + image
        input = caffe.io.load_image(pic)       
        image_show =cv2.imread(pic)  
        result = det(input,transformer,net)       
        #print(image_show.shape)
        #print(image, result)
        res = vis_detections(image_show,result)
        cv2.imwrite('./det_res/'+ image, res)
        
        
def parse_args():
    parser = argparse.ArgumentParser()
    '''parse args'''
    parser.add_argument('--image_dir', default='./det_imgs/')
    parser.add_argument('--model_def', default='./detection/deploy.prototxt')
    parser.add_argument('--model_weights', default='./detection/baidu_iter_140000.caffemodel')
    
    return parser.parse_args()
    
if __name__ == '__main__':
    main(parse_args())
