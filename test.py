#!/usr/bin/env python

import numpy as np
import os.path
import tensorflow as tf
import cv2
import glob

from PIL import Image
from utils import label_map_util
from utils import visualization_utils as vis_util

set_num = 5

my_dir = os.path.abspath(os.path.dirname(__file__))
PATH_TO_CKPT = os.path.join(
    my_dir, "trained-inference-graphs/train%d/frozen_inference_graph.pb"%set_num)
PATH_TO_LABELS = os.path.join(my_dir, "annotations/label_map.pbtxt")
NUM_CLASSES = 1

detection_graph = tf.Graph()
session_config = tf.ConfigProto()
session_config.gpu_options.allow_growth = True
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

sess = tf.Session(graph=detection_graph,
                  config=session_config)

# Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


files = glob.glob('images/test%d/*.jpg'%set_num)
for i, img_file in enumerate(files):
    image_np = cv2.imread(img_file)
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name(
        'image_tensor:0')
    boxes = detection_graph.get_tensor_by_name(
        'detection_boxes:0')
    scores = detection_graph.get_tensor_by_name(
        'detection_scores:0')
    classes = detection_graph.get_tensor_by_name(
        'detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name(
        'num_detections:0')
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8)

    conf_list = np.squeeze(scores).astype(str)
    box_list = np.squeeze(boxes)
    predictions = []
    boxes = []
    num = int(num_detections[0])
    file_name = os.path.splitext(os.path.basename(img_file))
    file = open('detections/set%d/'%set_num+file_name[0]+'.txt','w')
    print('\n'+file_name[0])
    for i in range(num):
        pred_boxpts = box_list[i]
        xmin = str(int(pred_boxpts[1]*image_np.shape[1]))
        ymin = str(int(pred_boxpts[0]*image_np.shape[0]))
        xmax = str(int(pred_boxpts[3]*image_np.shape[1]))
        ymax = str(int(pred_boxpts[2]*image_np.shape[0]))
        print('boat'+' '+conf_list[i]+' '+xmin+' '+ymin+' '+xmax+' '+ymax)
        file.write('boat'+' '+conf_list[i]+' '+xmin+' '+ymin+' '+xmax+' '+ymax+'\n')
    file.close()


    cv2.imshow('test', image_np)
    cv2.waitKey(1)


cv2.destroyAllWindows()
sess.close()
