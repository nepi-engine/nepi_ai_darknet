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
import cv2
import numpy as np

import darknet

from nepi_sdk import nepi_ros
from nepi_sdk import nepi_utils
from nepi_sdk import nepi_img
from nepi_sdk import nepi_ais

from nepi_api.ai_if_detector import AiDetectorIF
from nepi_api.messages_if import MsgIF


# Define your PyTorch model and load the weights
# model = ...



class Yolov3Detector():
    default_config_dict = {'threshold': 0.3,'max_rate': 5}
    #######################
    ### Node Initialization
    DEFAULT_NODE_NAME = "ai_yolov3" # Can be overwitten by luanch command
    def __init__(self):
        ####  NODE Initialization ####
        nepi_ros.init_node(name= self.DEFAULT_NODE_NAME)
        self.class_name = type(self).__name__
        self.base_namespace = nepi_ros.get_base_namespace()
        self.node_name = nepi_ros.get_node_name()
        self.node_namespace = nepi_ros.get_node_namespace()

        ##############################  
        # Create Msg Class
        self.msg_if = MsgIF(log_name = self.class_name)
        self.msg_if.pub_info("Starting Node Initialization Processes")

        ##############################  
        # Initialize Class Variables
        node_params = nepi_ros.get_param("~")
        self.msg_if.pub_info("Starting node params: " + str(node_params))
        self.all_namespace = nepi_ros.get_param("~all_namespace","")
        if self.all_namespace == "":
            self.all_namespace = self.node_namespace
        self.weight_file_path = nepi_ros.get_param("~weight_file_path","")
        self.config_file_path = nepi_ros.get_param("~config_file_path","")
        if self.config_file_path == "" or self.weight_file_path == "":
            self.msg_if.pub_warn("Failed to get required node info from param server: ")
            nepi_ros.signal_shutdown("Failed to get valid model info from param")
        else:
            # The ai_models param is created by the launch files load network_param_file line
            model_info = nepi_ros.get_param("~ai_model","")

            if model_info == "":
                self.msg_if.pub_warn("Failed to get required model info from params: ")
                nepi_ros.signal_shutdown("Failed to get valid model file paths")
            else:
                try: 
                    model_framework = model_info['framework']['name']
                    model_type = model_info['type']['name']
                    model_description = model_info['description']['name']
                    self.classes = model_info['classes']['names']
                    self.proc_img_width = model_info['image_size']['image_width']['value']
                    self.proc_img_height = model_info['image_size']['image_height']['value']
                except Exception as e:
                    self.msg_if.pub_warn("Failed to get required model info from params: " + str(e))
                    nepi_ros.signal_shutdown("Failed to get valid model file paths")

                if model_framework != 'yolov3':
                    self.msg_if.pub_warn("Model not a yolov3 model: " + model_framework)
                    nepi_ros.signal_shutdown("Model not a valid framework")

                self.msg_if.pub_info("Loading model: " + self.node_name)
                self.model = darknet.load_network(self.config_file_path, self.weight_file_path)

                #self.msg_if.pub_info("Waiting " + str(800) + " seconds for model to load")
                #nepi_ros.sleep(800)

                self.msg_if.pub_info("Starting ai_if with default_config_dict: " + str(self.default_config_dict))
                self.ai_if = AiDetectorIF(model_name = self.node_name,
                                    framework = model_framework,
                                    description = model_description,
                                    proc_img_height = self.proc_img_height,
                                    proc_img_width = self.proc_img_width,
                                    classes_list = self.classes,
                                    default_config_dict = self.default_config_dict,
                                    all_namespace = self.all_namespace,
                                    preprocessImageFunction = self.preprocessImage,
                                    processDetectionFunction = self.processDetection,
                                    has_img_tiling = False)

                #########################################################
                ## Initiation Complete
                self.msg_if.pub_info("Initialization Complete")
                # Spin forever (until object is detected)
                nepi_ros.spin()
                #########################################################        
              

    def preprocessImage(self,cv2_img,options_dict):
        height, width = cv2_img.shape[:2]
        
        # For Future
        '''
        tile = False
        if 'tile'  in options_dict.keys():
            tile = options_dict['tile']
        '''
        [cv2_img,ratio,new_width,new_height] = nepi_img.resize_proportionally(cv2_img, self.proc_img_width,self.proc_img_height,interp = cv2.INTER_NEAREST)
        
        # Convert BW image to RGB
        if nepi_img.is_gray(cv2_img):
            cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_GRAY2BGR)

        #self.msg_if.pub_info(":yolov5: Preprocessed image with image size: " + str(cv2_img.shape))
        # Create image dict with new image
        img_dict = dict()
        img_dict['cv2_img'] = cv2_img
        img_shape = cv2_img.shape
        img_dict['src_width'] = width 
        img_dict['src_height'] = height 
        img_dict['prc_width'] = new_width 
        img_dict['prc_height'] = new_height 
        img_dict['ratio'] = ratio 
        img_dict['tiling'] = False

        return img_dict



    def processDetection(self,img_dict, threshold):
        detect_dict_list = []
        tile = False
        # For Future
        #if 'tile'  in options_dict.keys():
        #    tile = options_dict['tile']
        if img_dict is not None:
            if 'cv2_img' in img_dict.keys():
                cv2_img = img_dict['cv2_img']
                if cv2_img is not None:

                    # Convert BGR image RGB
                    cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)

                    img_for_detect = darknet.make_image(self.proc_img_width, self.proc_img_height, 3)
                    darknet.copy_image_from_bytes(img_for_detect, cv2_img.tobytes())
                
                    # Run Detection
                    detections = darknet.detect_image(self.model, self.classes, img_for_detect, thresh=threshold)
                    #self.msg_if.pub_warn("Detections: " + str(detections))
                    rescale_ratio = float(1) / img_dict['ratio']
                    for label, confidence, bbox in detections:
                        det_name = label
                        det_id = self.classes.index(det_name)
                        det_prob = float(confidence) / 100.0
                        det_box = self.convert2original(cv2_img, bbox, self.proc_img_height, self.proc_img_width)
                        #det_box = self.convert4cropping(cv2_img, bbox, self.proc_img_height, self.proc_img_width)
                        detect_dict = {
                            'name': str(label), # Class String Name
                            'id': det_id, # Class Index from Classes List
                            'uid': '', # Reserved for unique tracking by downstream applications
                            'prob': det_prob, # Probability of detection
                            'xmin': det_box[0]-int(det_box[2]/2.),
                            'ymin': det_box[1]-int(det_box[3]/2.) ,
                            'xmax': det_box[0] + int(det_box[2]/2.),
                            'ymax': det_box[1] + int(det_box[3]/2.),
                            'width_pixels': cv2_img_width,
                            'height_pixels': cv2_img_height,
                            'area_pixels': det_box[2] * det_box[3],
                            'area_ratio': (det_box[2] * det_box[3]) / cv2_img_area
                        }
                        # Rescale to orig image size
                        detect_dict['xmin'] = int(detect_dict['xmin'] * rescale_ratio)
                        detect_dict['ymin'] = int(detect_dict['ymin'] * rescale_ratio)
                        detect_dict['xmax'] = int(detect_dict['xmax'] * rescale_ratio)
                        detect_dict['ymax'] = int(detect_dict['ymax'] * rescale_ratio)
                        detect_dict_list.append(detect_dict)
                        #self.msg_if.pub_info("Got detect dict entry: " + str(detect_dict))

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
