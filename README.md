Things to install/do before running:

Download the object_detection folder here: https://github.com/tensorflow/models/tree/master/research/object_detection
Download the model here: http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_oid_v4_2018_12_12.tar.gz
Open model and move model directory into the object_detection folder
Rename model directory to one indicated in the code


Things to change in the code before running:

Change all paths relevant to own situation. This includes:
   - Path to images to be cropped
   - Path to results directory

Change class indexes relevant to classes you want to detect. 
The label map can be found inside object_detection/data

Change object detection score threshold to relevant scores



