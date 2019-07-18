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

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.12.0'):
    raise ImportError('Please upgrade your TensorFlow installation to v1.12.*.')


from utils import label_map_util

from utils import visualization_utils as vis_util

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

MODEL_NAME = 'faster_rcnn_inception_resnet_v2_atrous_oid_v4'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = './oid_v4_label_map.pbtxt'

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
        
        name_list = ['ishaan','jessica','michael']
        for name in name_list:
            TEST_IMAGE_PATHS = glob.glob('./annotation_imgs/{}/*jpg'.format(name))
            TEST_IMAGE_PATHS.extend(glob.glob('./annotation_imgs/{}/*png'.format(name)))
            try:
                os.stat('./result/{}/'.format(name))
            except:
                os.makedirs('./result/{}/'.format(name))
            for image_path in tqdm.tqdm(TEST_IMAGE_PATHS):
                try:
                    image = Image.open(image_path)
                    if max(image.size)<1000:
                        image=image.resize((image.size[0]*2, image.size[1]*2), resample=Image.BILINEAR)
                    save_name = image_path.split('/')[-1]
                    # the array based representation of the image will be used later in order to prepare the
                    # result image with boxes and labels on it.
                    image_np = load_image_into_numpy_array(image)

                    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                    image_np_expanded = np.expand_dims(image_np, axis=0)
                except:
                    continue

                # Actual detection.
                output_dict = run_inference_for_single_image(image_np_expanded, sess)
                show_image = np.array(image) 

                # Convert RGB to BGR 
                show_image = show_image[:, :, ::-1].copy() 
                for index in range(len(output_dict['detection_scores'])):
                    if output_dict['detection_classes'][index]==211: #output_dict['detection_scores'][index]>0.0:# 
                        show_image=cv2.rectangle(show_image, 
                                (int(output_dict['detection_boxes'][index][1]*show_image.shape[1]),
                                int(output_dict['detection_boxes'][index][0]*show_image.shape[0])),
                                (int(output_dict['detection_boxes'][index][3]*show_image.shape[1]),
                                int(output_dict['detection_boxes'][index][2]*show_image.shape[0])),
                                (0,0,255), 2)
                cv2.imwrite('./result/{}/'.format(name)+save_name, show_image)
