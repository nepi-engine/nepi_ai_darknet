#!/usr/bin/env python
#
# Copyright (c) 2024 Numurus <https://www.numurus.com>.
#
# This file is part of nepi applications (nepi_apps) repo
# (see https://https://github.com/nepi-engine/nepi_apps)
#
# License: nepi applications are licensed under the "Numurus Software License", 
# which can be found at: <https://numurus.com/wp-content/uploads/Numurus-Software-License-Terms.pdf>
#
# Redistributions in source code must retain this top-level comment block.
# Plagiarizing this software to sidestep the license obligations is illegal.
#
# Contact Information:
# ====================
# - mailto:nepi@numurus.com
#

import os
import time
import copy
import sys
import rospy
import cv2
import numpy as np

import darknet

from nepi_sdk import nepi_ros
from nepi_sdk import nepi_msg
from nepi_sdk import nepi_ais

from nepi_sdk.ai_detector_if import AiDetectorIF

# Define your PyTorch model and load the weights
# model = ...


TEST_DETECTION_DICT_ENTRY = {
    'name': 'TEST_CLASS', # Class String Name
    'id': 1, # Class Index from Classes List
    'uid': '', # Reserved for unique tracking by downstream applications
    'prob': .3, # Probability of detection
    'xmin': 10,
    'ymin': 10,
    'xmax': 50,
    'ymax': 50,
    'width_pixels': 40,
    'height_pixels': 40,
    'area_pixels': 16000,
    'area_ratio': 0.22857
}



class Yolov3Detector():
    defualt_config_dict = {'threshold': 0.3,'max_rate': 5}
    #######################
    ### Node Initialization
    DEFAULT_NODE_NAME = "ai_yolov3" # Can be overwitten by luanch command
    def __init__(self):
        #### APP NODE INIT SETUP ####
        nepi_ros.init_node(name= self.DEFAULT_NODE_NAME)
        self.node_name = nepi_ros.get_node_name()
        self.base_namespace = nepi_ros.get_base_namespace()
        self.node_namespace = self.base_namespace + self.node_name
        nepi_msg.createMsgPublishers(self)
        nepi_msg.publishMsgInfo(self,"Starting Initialization Processes")
        ##############################
        # Initialize parameters and fields.
        node_params = nepi_ros.get_param(self,"~")
        nepi_msg.publishMsgInfo(self,"Starting node params: " + str(node_params))
        self.all_namespace = nepi_ros.get_param(self,"~all_namespace","")
        if self.all_namespace == "":
            self.all_namespace = self.node_namespace
        self.weight_file_path = nepi_ros.get_param(self,"~weight_file_path","")
        self.config_file_path = nepi_ros.get_param(self,"~config_file_path","")
        if self.config_file_path == "" or self.weight_file_path == "":
            nepi_msg.publishMsgWarn(self,"Failed to get required node info from param server: ")
            rospy.signal_shutdown("Failed to get valid model info from param")
        else:
            # The ai_models param is created by the launch files load network_param_file line
            model_info = nepi_ros.get_param(self,"~ai_model","")

            if model_info == "":
                nepi_msg.publishMsgWarn(self,"Failed to get required model info from params: ")
                rospy.signal_shutdown("Failed to get valid model file paths")
            else:
                try: 
                    model_framework = model_info['framework']['name']
                    model_type = model_info['type']['name']
                    model_description = model_info['description']['name']
                    self.classes = model_info['classes']['names']
                    self.model_img_width = model_info['image_size']['image_width']['value']
                    self.model_img_height = model_info['image_size']['image_height']['value']
                except Exception as e:
                    nepi_msg.publishMsgWarn(self,"Failed to get required model info from params: " + str(e))
                    rospy.signal_shutdown("Failed to get valid model file paths")

                if model_framework != 'yolov3':
                    nepi_msg.publishMsgWarn(self,"Model not a yolov3 model: " + model_framework)
                    rospy.signal_shutdown("Model not a valid framework")

                nepi_msg.publishMsgInfo(self,"Loading model: " + self.node_name)
                self.model = darknet.load_network(self.config_file_path, self.weight_file_path)

                #nepi_msg.publishMsgInfo(self,"Waiting " + str(800) + " seconds for model to load")
                #nepi_ros.sleep(800)

                nepi_msg.publishMsgInfo(self,"Starting ai_if with defualt_config_dict: " + str(self.defualt_config_dict))
                self.ai_if = AiDetectorIF(model_name = self.node_name,
                                    framework = model_framework,
                                    description = model_description,
                                    img_height = self.model_img_height,
                                    img_width = self.model_img_width,
                                    classes_list = self.classes,
                                    defualt_config_dict = self.defualt_config_dict,
                                    all_namespace = self.all_namespace,
                                    processDetectionFunction = self.processDetection)

                #########################################################
                ## Initiation Complete
                nepi_msg.publishMsgInfo(self,"Initialization Complete")
                # Spin forever (until object is detected)
                nepi_ros.spin()
                #########################################################        
              



    def processDetection(self,cv2_img, threshold):
        #detect_dict_list = [TEST_DETECTION_DICT_ENTRY]
        cv2_shape = cv2_img.shape
        cv2_img_width = cv2_shape[1] 
        cv2_img_height = cv2_shape[0] 
        cv2_img_area = cv2_img_width * cv2_img_height
        # Convert the image
        prev_time = time.time()
        frame_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (self.model_img_width, self.model_img_height),
                                   interpolation=cv2.INTER_LINEAR)
        img_for_detect = darknet.make_image(self.model_img_width, self.model_img_height, 3)
        darknet.copy_image_from_bytes(img_for_detect, frame_resized.tobytes())
        convert_time = (time.time() - prev_time)
        #nepi_msg.publishMsgInfo(self,"Convet Time: {:.2f}".format(convert_time))        

        # Run Detection
        prev_time = time.time()
        detections = darknet.detect_image(self.model, self.classes, img_for_detect, thresh=threshold)
        detect_time = (time.time() - prev_time)
        #nepi_msg.publishMsgInfo(self,"Detect Time: {:.2f}".format(detect_time))
        #nepi_msg.publishMsgInfo(self,"Detections: " + str(detections))
        detect_dict_list = []
        for label, confidence, bbox in detections:
            det_name = label
            det_id = self.classes.index(det_name)
            det_prob = float(confidence) / 100.0
            #det_box = self.convert2original(cv2_img, bbox, self.model_img_height, self.model_img_width)
            det_box = self.convert4cropping(cv2_img, bbox, self.model_img_height, self.model_img_width)
            detect_dict = {
                'name': str(label), # Class String Name
                'id': det_id, # Class Index from Classes List
                'uid': '', # Reserved for unique tracking by downstream applications
                'prob': det_prob, # Probability of detection
                'xmin': det_box[0]-int(det_box[2]/2),
                'ymin': det_box[1]-int(det_box[3]/2) ,
                'xmax': det_box[0] + int(det_box[2]/2),
                'ymax': det_box[1] + int(det_box[3]/2),
                'width_pixels': cv2_img_width,
                'height_pixels': cv2_img_height,
                'area_pixels': det_box[2] * det_box[3],
                'area_ratio': (det_box[2] * det_box[3]) / cv2_img_area,
            }
            detect_dict_list.append(detect_dict)
            #nepi_msg.publishMsgInfo(self,"Got detect dict entry: " + str(detect_dict))

        return detect_dict_list



    def convert2relative(self,bbox, preproc_h, preproc_w):
        """
        YOLO format use relative coordinates for annotation
        """
        x, y, w, h = bbox
        return x / preproc_w, y / preproc_h, w / preproc_w, h / preproc_h


    def convert2original(self,image, bbox, preproc_h, preproc_w):
        x, y, w, h = self.convert2relative(bbox, preproc_h, preproc_w)

        image_h, image_w, __ = image.shape

        orig_x = int(x * image_w)
        orig_y = int(y * image_h)
        orig_width = int(w * image_w)
        orig_height = int(h * image_h)

        bbox_converted = (orig_x, orig_y, orig_width, orig_height)

        return bbox_converted


    # @TODO - cfati: Unused
    def convert4cropping(self,image, bbox, preproc_h, preproc_w):
        x, y, w, h = self.convert2relative(bbox, preproc_h, preproc_w)

        image_h, image_w, __ = image.shape

        orig_left = int((x - w / 2.) * image_w)
        orig_right = int((x + w / 2.) * image_w)
        orig_top = int((y - h / 2.) * image_h)
        orig_bottom = int((y + h / 2.) * image_h)

        if orig_left < 0:
            orig_left = 0
        if orig_right > image_w - 1:
            orig_right = image_w - 1
        if orig_top < 0:
            orig_top = 0
        if orig_bottom > image_h - 1:
            orig_bottom = image_h - 1

        bbox_cropping = (orig_left, orig_top, orig_right, orig_bottom)

        return bbox_cropping



if __name__ == '__main__':
    Yolov3Detector()
