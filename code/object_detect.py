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
# Download the object_detection directory here: https://github.com/tensorflow/models/tree/master/research/object_detection
sys.path.insert(0, "/home/michael/tensorflow/research/")
from object_detection.utils import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.12.0'):
    raise ImportError('Please upgrade your TensorFlow installation to v1.12.*.')


from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

#Path to object_detection directory
PATH_TO_OD_FOLDER = '/home/michael/tensorflow/research/object_detection/'

#Donwload the model here: http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_oid_v4_2018_12_12.tar.gz
#Rename model to name below, and move into object_detection directory
MODEL_NAME = PATH_TO_OD_FOLDER + 'faster_rcnn_inception_resnet_v2_atrous_oid_v4'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = PATH_TO_OD_FOLDER + 'data/oid_v4_label_map.pbtxt'

#List of classes that we want to draw bounding boxes on. Info on index number to class mapping can be found in file
#Given by PATH_TO_LABELS
RELEVANT_CLASS_LIST = [211]#[34, 39, 69, 87, 98, 139, 161, 167, 177, 211, 216, 221, 228, 250, 292, 308, 
                    #351, 365, 366, 399, 408, 433, 502, 503, 540, 573, 601] 
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

#Font variables for the strings
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.4
THICKNESS = 1

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
        
        #Change path to directory containing the images to be run on. Change this path for all glob.glob functions too
        name_list = glob.glob('./data/annotation_imgs/**/')
        for n in name_list:
            name = n.split('/')[3]
            TEST_IMAGE_PATHS = glob.glob('./data/annotation_imgs_full/{}/*jpg'.format(name))
            TEST_IMAGE_PATHS.extend(glob.glob('./data/annotation_imgs_full/{}/*png'.format(name)))            
            TEST_IMAGE_PATHS.extend(glob.glob('./data/annotation_imgs_full/{}/*gif'.format(name)))
            TEST_IMAGE_PATHS.extend(glob.glob('./data/annotation_imgs_full/{}/*jpeg'.format(name)))
            print(TEST_IMAGE_PATHS)
            
            #Creates results in a subdirectory inside a data directory (If you change this path, change for all paths below too)
            try:
                os.stat('./data/result/{}/'.format(name))
            except:
                os.makedirs('./data/result/{}/'.format(name))
            for image_path in tqdm.tqdm(TEST_IMAGE_PATHS):
                try:
                    image = Image.open(image_path)
                    if max(image.size)<1000:
                        image=image.resize((image.size[0]*2, image.size[1]*2), resample=Image.BILINEAR)
                    save_name = image_path.split('/')[-1]
                    # the array based representation of the image will be used later in order to prepare the
                    # result image with boxes and labels on it.

                    # error occuring for some images when loading inmage to numpy array
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
                class_name = 'none'
                final_image = original_image.copy()
                class_image = original_image.copy()

                for class_index in RELEVANT_CLASS_LIST:
                    # Iterates through relevant detection thresholds
                    for threshold in [0.1, 0.15, 0.2, 0.3]:
                        thresholdstr = str(threshold)
                        for index in range(output_dict['num_detections']):
                            if output_dict['detection_classes'][index] == class_index and output_dict['detection_scores'][index] > threshold: 
                                RELEVANT_CLASS_DICT[class_index] = True
                                for show_image in [class_image, final_image]:
                                    (left, right, top, bottom) =  (int(output_dict['detection_boxes'][index][1]*show_image.shape[1]),
                                             int(output_dict['detection_boxes'][index][3]*show_image.shape[1]),
                                             int(output_dict['detection_boxes'][index][0]*show_image.shape[0]),
                                             int(output_dict['detection_boxes'][index][2]*show_image.shape[0])) 

                                    cv2.rectangle(show_image, (left, top),(right, bottom), (0,0,255), 2)

                                    #Write class names (code from visualization_utils in object_detection.utils)
                                    if output_dict['detection_classes'][index] in category_index.keys():
                                        class_name = category_index[output_dict['detection_classes'][index]]['name']
                                    display_str = str(class_name)

                                    text_size, baseline= cv2.getTextSize(display_str, FONT, FONT_SCALE, THICKNESS)
                                    text_width = text_size[0]
                                    text_height = text_size[1]
                                    display_str_height = (1 + 2 * 0.05) * text_height
                                    
                                    if top - display_str_height > 0:
                                        text_bottom = top
                                    else:
                                        text_bottom = bottom + display_str_height + baseline
             
                                    cv2.rectangle(show_image, (left, int(text_bottom - text_height - baseline)), 
                                                 (int(left + text_width), int(text_bottom)), (0, 0, 255), -1)
                                    cv2.putText(show_image, display_str, (left, int(text_bottom - baseline)), 
                                               FONT, FONT_SCALE, (255,255,255), THICKNESS, True)

                        if RELEVANT_CLASS_DICT[class_index]:  
                            try:
                                os.stat('./data/result/{}/{}/{}/'.format(name, thresholdstr, class_name))
                            except:
                                os.makedirs('./data/result/{}/{}/{}/'.format(name, thresholdstr, class_name))

                            cv2.imwrite('./data/result/{}/{}/{}/'.format(name, thresholdstr, class_name)+save_name, class_image)
                            class_image = original_image.copy()

                    try:
                        os.stat('./data/result/{}/{}/all_classes/'.format(name, thresholdstr))
                    except:
                        os.makedirs('./data/result/{}/{}/all_classes/'.format(name, thresholdstr))
                            
                    cv2.imwrite('./data/result/{}/{}/all_classes/'.format(name, thresholdstr)+save_name, final_image)

                    for i in RELEVANT_CLASS_LIST:
                        RELEVANT_CLASS_DICT[i] = False
