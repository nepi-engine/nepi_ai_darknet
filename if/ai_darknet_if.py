#!/usr/bin/env python

import sys
import os
import os.path

import glob
import subprocess
import yaml
import time
import rospy
import numpy as np


from nepi_edge_sdk_base import nepi_ros


from std_msgs.msg import Empty, Float32
from nepi_ros_interfaces.msg import ObjectCount
from nepi_ros_interfaces.srv import ImageClassifierStatusQuery, ImageClassifierStatusQueryResponse


from nepi_edge_sdk_base.save_cfg_if import SaveCfgIF


AI_NAME = 'Darknet' # Use in display menus
FILE_TYPE = 'AIF_IF'


class DarknetAIF(object):
    FIXED_LOADING_START_UP_TIME_S = 5.0 # Total guess
    ESTIMATED_WEIGHT_LOAD_RATE_BYTES_PER_SECOND = 16000000.0 # Very roughly empirical based on YOLOv3

    def __init__(self, ai_dict, pub_sub_namespace, models_lib_path):
      #rospy.logwarn("Darknet IF got ai_dict: " + str(ai_dict))
      if pub_sub_namespace[-1] == "/":
        pub_sub_namespace = pub_sub_namespace[:-1]
      self.pub_sub_namespace = pub_sub_namespace
      self.models_lib_path = models_lib_path
      self.pkg_name = ai_dict['pkg_name']
      self.node_name = ai_dict['node_name']
      self.node_file = ai_dict['node_file_name']
      self.launch_pkg = ai_dict['launch_pkg_name']
      self.launch_file = ai_dict['launch_file_name']
      self.model_prefix = ai_dict['model_prefix']
      self.models_folder = ai_dict['models_folder_name']
      self.models_folder_path =  os.path.join(self.models_lib_path, self.models_folder)
      #rospy.logwarn("Darknet models path: " + self.models_folder_path)

    
    #################
    # Darknet Model Functions

    def getModelsDict(self):
        models_dict = dict()
        classifier_name_list = []
        classifier_size_list = []
        classifier_classes_list = []
        # Try to obtain the path to Darknet models from the system_mgr
        cfg_path_config_folder = os.path.join(self.models_folder_path, 'config')
        rospy.loginfo("Darknet looking for models config files in folder: " + cfg_path_config_folder)
        # Grab the list of all existing darknet cfg files
        if os.path.exists(cfg_path_config_folder) == False:
            rospy.loginfo("Yolov5: Failed to find models config files in folder: " + cfg_path_config_folder)
            return models_dict
        else:
            self.cfg_files = glob.glob(os.path.join(cfg_path_config_folder,'*.yaml'))
            # Remove the ros.yaml file -- that one doesn't represent a selectable trained neural net
            try:
                self.cfg_files.remove(os.path.join(cfg_path_config_folder,'ros.yaml'))
            except:
                rospy.logwarn("Unexpected: ros.yaml is missing from the darknet config path " + cfg_path_config_folder)

            for f in self.cfg_files:
                yaml_stream = open(f, 'r')
                # Validate that it is a proper config file and gather weights file size info for load-time estimates
                cfg_dict = yaml.load(yaml_stream)
                #rospy.logwarn("" + str(cfg_dict))
                
                yaml_stream.close()
                if ("yolo_model" not in cfg_dict) or ("weight_file" not in cfg_dict["yolo_model"]) or ("name" not in cfg_dict["yolo_model"]["weight_file"]):
                    rospy.logwarn("Debug: " + str(cfg_dict))
                    rospy.logwarn("File does not appear to be a valid A/I model config file: " + f + "... not adding this classifier")
                    continue


                classifier_name = os.path.splitext(os.path.basename(f))[0]
                weight_file = os.path.join(self.models_folder_path , "yolo_network_config", "weights",cfg_dict["yolo_model"]["weight_file"]["name"])
                if not os.path.exists(weight_file):
                    rospy.logwarn("Classifier " + classifier_name + " specifies non-existent weights file " + weight_file + "... not adding this classifier")
                    continue
                classifier_keys = list(cfg_dict.keys())
                classifier_key = classifier_keys[0]
                classifier_classes_list.append(cfg_dict[classifier_key]['detection_classes']['names'])
                #rospy.logwarn("classes: " + str(classifier_classes_list))
                classifier_name_list.append(classifier_name)
                classifier_size_list.append(os.path.getsize(weight_file))
            for i,name in enumerate(classifier_name_list):
                model_name = self.model_prefix + name
                model_dict = dict()
                model_dict['name'] = name
                model_dict['size'] = classifier_size_list[i]
                load_time = self.FIXED_LOADING_START_UP_TIME_S + (classifier_size_list[i] / self.ESTIMATED_WEIGHT_LOAD_RATE_BYTES_PER_SECOND)
                model_dict['load_time'] =  load_time
                model_dict['classes'] = classifier_classes_list[i]
                models_dict[model_name] = model_dict
            #rospy.logwarn("Classifier returning models dict" + str(models_dict))
            return models_dict


    def startClassifier(self, classifier, source_img_topic, threshold):
        # Build Darknet new classifier launch command

        launch_cmd_line = [
            "roslaunch", self.launch_pkg, self.launch_file,
            "pkg_name:=" + self.launch_pkg,
            "namespace:=" + self.pub_sub_namespace, 
            "node_name:=" + self.node_name,
            "node_file:=" + self.node_file,
            "yolo_weights_path:=" + os.path.join(self.models_folder_path, "yolo_network_config/weights"),
            "yolo_config_path:=" + os.path.join(self.models_folder_path, "yolo_network_config/cfg"),
            "ros_param_file:=" + os.path.join(self.models_folder_path, "config/ros.yaml"),
            "network_param_file:=" + os.path.join(self.models_folder_path, "config", classifier + ".yaml"),
            "input_img:=" + source_img_topic,
            "detection_threshold:=" + str(threshold)
        ]
        
        rospy.loginfo("Launching Darknet ROS Process: " + str(launch_cmd_line))
        self.ros_process = subprocess.Popen(launch_cmd_line)
        

        # Setup Classifier Setup Tracking Progress




    def stopClassifier(self):
        rospy.loginfo("Stopping classifier")
        if not (None == self.ros_process):
            self.ros_process.terminate()
            self.ros_process = None
        self.current_classifier = "None"
        self.current_img_topic = "None"
        
        #self.current_threshold = 0.3

    

if __name__ == '__main__':
    DarknetAIF()
