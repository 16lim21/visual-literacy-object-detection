import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import glob
import tqdm

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

import cv2

# Append the path to the directory containing the object_detection directory
sys.path.insert(0, "/home/michael/tensorflow/research/")

from object_detection.utils import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.12.0'):
        raise ImportError('Please upgrade your TensorFlow installation to v1.12.*.')

from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

#Absolute path to object_detection folder
PATH_TO_OD_FOLDER = '/home/michael/tensorflow/research/object_detection/'
MODEL_NAME = PATH_TO_OD_FOLDER + 'faster_rcnn_inception_resnet_v2_atrous_oid_v4'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = PATH_TO_OD_FOLDER + 'data/oid_v4_label_map.pbtxt'

#List of classes that we want to draw bounding boxes on
RELEVANT_CLASS_LIST = [211] #only focusing on flags right now 
RELEVANT_CLASS_DICT = {}

for i in RELEVANT_CLASS_LIST:
    RELEVANT_CLASS_DICT[i] = False

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

# Size, in inches, of the output images.
IMAGE_SIZE = (18, 12)

def run_inference_for_single_image(image, sess):
    image_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name('image_tensor:0')

    # Run inference
    output_dict = sess.run(tensor_dict,
                           feed_dict={image_tensor: image})

    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict[
        'detection_classes'][0].astype(np.int64)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]

    return output_dict

with detection_graph.as_default():
    with tf.compat.v1.Session() as sess:
        # Get handles to input and output tensors
        ops = tf.compat.v1.get_default_graph().get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        tensor_dict = {}
        #, 'detection_masks'
        for key in [
            'num_detections', 'detection_boxes', 'detection_scores',
            'detection_classes']:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
                tensor_dict[key] = tf.compat.v1.get_default_graph().get_tensor_by_name(
                                    tensor_name)
       
        #List of all subdirectories in the directory containing all images
        name_list = glob.glob('./data/images/**/')
        total_flags = 0
        for n in name_list:
            folder_flags = 0
            #Indexed number depends on folder location
            name = n.split('/')[3]

            #Find paths of all images inside the image directory
            TEST_IMAGE_PATHS = glob.glob('./data/images/{}/**/*jpg'.format(name), recursive=True)
            TEST_IMAGE_PATHS.extend(glob.glob('./data/images/{}/**/*png'.format(name), recursive=True))
            TEST_IMAGE_PATHS.extend(glob.glob('./data/images/{}/**/*gif'.format(name), recursive=True))
            TEST_IMAGE_PATHS.extend(glob.glob('./data/images/{}/**/*jpeg'.format(name), recursive=True))
            
            #Creates new directory to store all cropped images
            try:
                os.stat('./data/crop/{}/'.format(name))
            except:
                os.makedirs('./data/crop/{}/'.format(name))
            for image_path in tqdm.tqdm(TEST_IMAGE_PATHS):
                try:
                    image = Image.open(image_path)
                    if max(image.size)<1000:
                        image=image.resize((image.size[0]*2, image.size[1]*2), resample=Image.BILINEAR)
                    save_full_name = image_path.split('/')[-1]
                    save_name, extension = save_full_name.split('.')
                    
                    # the array based representation of the image will be used later in order to prepare the
                    # result image with boxes and labels on it.
                    image_np = load_image_into_numpy_array(image)

                    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                    image_np_expanded = np.expand_dims(image_np, axis=0)
                except:
                    print('skipped {}'.format(image_path))
                    continue

                # Actual detection.
                output_dict = run_inference_for_single_image(image_np_expanded, sess)
                original_image = np.array(image) 

                # Convert RGB to BGR 
                original_image = original_image[:, :, ::-1].copy()
                
                crop_num = 0

                #Check for classes that are relevant to us
                for class_index in RELEVANT_CLASS_LIST:
                    #iterate through all detections found in the image
                    for index in range(output_dict['num_detections']): 
                        #if detection in question is relevant to us, crop it
                        #Change detection score threshold for relevant use
                        if output_dict['detection_classes'][index] == class_index and output_dict['detection_scores'][index] > 0.15: 
                            RELEVANT_CLASS_DICT[class_index] = True

                            #Number at end allows images to be cropped slightly larger than detection_boxes. Change according to own use
                            (left, right, top, bottom) =  (int(output_dict['detection_boxes'][index][1]*original_image.shape[1]*0.9),
                                     int(output_dict['detection_boxes'][index][3]*original_image.shape[1]*1.1),
                                     int(output_dict['detection_boxes'][index][0]*original_image.shape[0]*0.9),
                                     int(output_dict['detection_boxes'][index][2]*original_image.shape[0]*1.1)) 

                            crop_image = original_image[top:bottom, left:right].copy()

                            try:
                                os.stat('./data/crop/{}/'.format(name))
                            except:
                                os.makedirs('./data/crop/{}/'.format(name))
                            
                            if name == 'Others':
                                subdir = image_path.split('/')[4]
                                try:
                                    os.stat('./data/crop/{}/{}/'.format(name,subdir))
                                except:
                                    os.makedirs('./data/crop/{}/{}/'.format(name,subdir))

                                cv2.imwrite('./data/crop/{}/{}/{}_{:05d}.{}'.format(name,subdir,save_name,crop_num,extension), crop_image)
                            else:
                                cv2.imwrite('./data/crop/{}/{}_{:05d}.{}'.format(name,save_name,crop_num,extension), crop_image)

                            crop_num += 1
                            total_flags += 1
                            folder_flags += 1

                for i in RELEVANT_CLASS_LIST:
                    RELEVANT_CLASS_DICT[i] = False

            print(folder_flags)
        print(total_flags)
