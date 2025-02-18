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

import sys
import os
import os.path
# ROS namespace setup for stand alone testing. Comment out for deployed version
#NEPI_BASE_NAMESPACE = '/nepi/s2x/'
#os.environ["ROS_NAMESPACE"] = NEPI_BASE_NAMESPACE[0:-1] # remove to run as automation script
import rospy

import glob
import subprocess
import yaml
import time
import numpy as np


from nepi_sdk import nepi_ros
from nepi_sdk import nepi_msg


from std_msgs.msg import Empty, Float32, Int32, String, Bool

from nepi_ros_interfaces.srv import SetBoolState, SetBoolStateRequest

from nepi_sdk.save_cfg_if import SaveCfgIF


TEST_AI_DICT = {
'description': 'Darknet ai framework support', 
'pkg_name': 'nepi_aif_darknet', 
'if_file_name': 'aif_darknet_if.py', 
'if_path_name': '/opt/nepi/ros/share/nepi_aifs', 
'if_module_name': 'aif_darknet_if', 
'if_class_name': 'DarknetAIF', 
'models_folder_name': 'darknet', 
'model_prefix': 'ai_darknet_', 
'launch_pkg_name': 'nepi_ai_darknet',
'launch_file_name': 'darknet_ros.launch', 
'node_file_name': 'nepi_ai_darknet_node.py',  
'active': True
}

TEST_LAUNCH_NAMESPACE = "/nepi/darknet_test"
TEST_MGR_NAMESPACE = "/nepi/ai_detector_mgr"
TEST_MODELS_LIB_PATH = "/mnt/nepi_storage/ai_models/"



class DarknetAIF(object):
    TYPICAL_LOAD_TIME_PER_MB = 3.5

    darknet_path = '/opt/nepi/ros/share/darknet'


    ai_node_dict = dict()
    def __init__(self, ai_dict,launch_namespace, mgr_namespace, models_lib_path):
      if launch_namespace[-1] == "/":
        launch_namespace = launch_namespace[:-1]
      self.launch_namespace = launch_namespace  
      #nepi_msg.printMsgWarn("Launch Namespace: " + self.launch_namespace)
      if mgr_namespace[-1] == "/":
        mgr_namespace = mgr_namespace[:-1]
      self.mgr_namespace = mgr_namespace
      self.models_lib_path = models_lib_path
      self.pkg_name = ai_dict['pkg_name']

      self.launch_pkg = ai_dict['launch_pkg_name']
      self.launch_file = ai_dict['launch_file_name']
      self.model_prefix = ai_dict['model_prefix']
      self.models_folder = ai_dict['models_folder_name']
      self.models_folder_path =  os.path.join(self.models_lib_path, self.models_folder)
      nepi_msg.printMsgInfo("Darknet models path: " + self.models_folder_path)

    
    #################
    # Darknet Model Functions

    def getModelsDict(self):
        models_dict = dict()
        # Try to obtain the path to Darknet models from the system_mgr
        nepi_msg.printMsgInfo("ai_darknet_if: Looking for model files in folder: " + self.models_folder_path)
        # Grab the list of all existing darknet cfg files
        if os.path.exists(self.models_folder_path) == False:
            nepi_msg.printMsgInfo("ai_darknet_if: Failed to find models folder: " + self.models_folder_path)
            return models_dict
        else:
            self.cfg_files = glob.glob(os.path.join(self.models_folder_path,'*.yaml'))
            nepi_msg.printMsgInfo("ai_darknet_if: Found network config files: " + str(self.cfg_files))
            # Remove the ros.yaml file -- that one doesn't represent a selectable trained neural net
            for f in self.cfg_files:
                cfg_dict = dict()
                success = False
                try:
                    #nepi_msg.printMsgWarn("ai_darknet_if: Opening yaml file: " + f) 
                    yaml_stream = open(f, 'r')
                    success = True
                    #nepi_msg.printMsgWarn("ai_darknet_if: Opened yaml file: " + f) 
                except Exception as e:
                    nepi_msg.printMsgWarn("ai_darknet_if: Failed to open yaml file: " + str(e))
                if success:
                    try:
                        # Validate that it is a proper config file and gather weights file size info for load-time estimates
                        #nepi_msg.printMsgWarn("ai_darknet_if: Loading yaml data from file: " + f) 
                        cfg_dict = yaml.load(yaml_stream)  
                        model_keys = list(cfg_dict.keys())
                        model_key = model_keys[0]
                        #nepi_msg.printMsgWarn("ai_darknet_if: Loaded yaml data from file: " + f) 
                    except Exception as e:
                        nepi_msg.printMsgWarn("ai_darknet_if: Failed load yaml data: " + str(e)) 
                        success = False 
                try: 
                    #nepi_msg.printMsgWarn("ai_darknet_if: Closing yaml data stream for file: " + f) 
                    yaml_stream.close()
                except Exception as e:
                    nepi_msg.printMsgWarn("ai_darknet_if: Failed close yaml file: " + str(e))
                
                if success == False:
                    nepi_msg.printMsgWarn("ai_darknet_if: File does not appear to be a valid A/I model param file: " + f + "... not adding this model")
                    continue
                #nepi_msg.printMsgWarn("ai_darknet_if: Import success: " + str(success) + " with cfg_dict " + str(cfg_dict))
                cfg_dict_keys = cfg_dict[model_key].keys()
                #nepi_msg.printMsgWarn("ai_darknet_if: Imported model key names: " + str(cfg_dict_keys))
                if ("config_file" not in cfg_dict_keys):
                    nepi_msg.printMsgWarn("ai_darknet_if: File does not specify a config file: " + f + "... not adding this model")
                    continue
                #nepi_msg.printMsgWarn("ai_darknet_if: Imported model key names: " + str(cfg_dict_keys))
                if ("weight_file" not in cfg_dict_keys):
                    nepi_msg.printMsgWarn("ai_darknet_if: File does not specify a weight file: " + f + "... not adding this model")
                    continue
                #nepi_msg.printMsgWarn("ai_darknet_if: Imported model key names: " + str(cfg_dict_keys))
                if ("image_size" not in cfg_dict_keys):
                    nepi_msg.printMsgWarn("ai_darknet_if: File does not specify a image size: " + f + "... not adding this model")
                    continue
                #nepi_msg.printMsgWarn("ai_darknet_if: Imported model key names: " + str(cfg_dict_keys))
                if ("detection_classes" not in cfg_dict_keys):
                    nepi_msg.printMsgWarn("ai_darknet_if: File does not specify a detection_classes: " + f + "... not adding this model")
                    continue

                param_file = os.path.basename(f)
                model_name = self.model_prefix + os.path.splitext(param_file)[0]
                config_file = cfg_dict[model_key]["config_file"]["name"]
                weight_file = cfg_dict[model_key]["weight_file"]["name"]
                
                #nepi_msg.printMsgWarn("ai_darknet_if: Checking that model weigths file exists: " + weigth_file_path + " for model name " + model_name)
                if not os.path.exists(weigth_file_path):
                    nepi_msg.printMsgWarn("ai_darknet_if: Classifier " + model_name + " specifies non-existent weights file " + weigth_file_path + "... not adding this model")
                    continue
                model_size = os.path.getsize(weigth_file_path)
                model_dict = dict()
                model_dict['model_name'] = model_name
                model_dict['model_path'] = self.models_folder_path
                model_dict['param_file'] = param_file
                model_dict['config_file'] = config_file
                model_dict['weight_file']= weight_file
                model_dict['size'] = model_size
                model_dict['load_time'] = self.TYPICAL_LOAD_TIME_PER_MB * model_size / 1000000
                model_dict['classes'] = cfg_dict[model_key]['detection_classes']['names']
                nepi_msg.printMsgInfo("ai_darknet_if: Model dict create for model : " + model_name)
                models_dict[model_name] = model_dict
            #nepi_msg.printMsgWarn("Classifier returning models dict" + str(models_dict))
        return models_dict


    def loadClassifier(self, model_dict):
        success = False
        model_name = model_dict['model_name']
        node_name = model_name
        node_namespace = os.path.join(self.launch_namespace, node_name)
        # Build Darknet new model_name launch command
        launch_cmd_line = [
            "roslaunch", self.launch_pkg, self.launch_file,
            "pkg_name:=" + self.launch_pkg,
            "node_name:=" + node_name,
            "node_namespace:=" + self.launch_namespace,
            "mgr_namespace:=" + self.mgr_namespace, 
            "param_file_path:=" + os.path.join(model_dict['model_path'],model_dict['param_file']),
            "config_path_path:=" + os.path.join(model_dict['model_path'],model_dict['config_file']),
            "weight_file_path:=" + os.path.join(model_dict['model_path'],model_dict['weight_file'])
        ]
        nepi_msg.printMsgInfo("ai_darknet_if: Launching Darknet AI node " + model_name + " with commands: " + str(launch_cmd_line))
        node_process = subprocess.Popen(launch_cmd_line)
        node_enable_srv_namespace = os.path.join(node_namespace,'set_enable')
        set_enable_service = rospy.ServiceProxy(node_enable_srv_namespace, SetBoolState)
        self.ai_node_dict[model_name] = {'namesapce':node_namespace, 'enable_srv': set_enable_service, 'process':node_process}
        success = True
        return success, node_namespace


    def killClassifier(self,model_name):
        if model_name in self.ai_node_dict.keys():
            node_process = self.ai_node_dict[model_name]['proceess']
            nepi_msg.printMsgInfo("ai_darknet_if: Killing Darknet AI node: " + model_name)
            if not (None == self.node_process):
                self.node_process.terminate()
            del self.ai_node_dict[model_name]



    def enableClassifier(self, model_name, val):
        if model_name in self.ai_node_dict.keys():
            nepi_msg.printMsgInfo("ai_darknet_if: Sending enable service request to model " + model_name)
            set_enable_service = self.ai_node_dict[model_name]['enable_srv']
            #nepi_msg.printMsgInfo("ai_darknet_if: Setting Darknet AI node " + model_name + " enbable to: " + str(val))
            success = False
            if not rospy.is_shutdown():
                try:
                    req = SetBoolStateRequest()
                    req.req_state = val
                    set_enable_response = set_enable_service(req)
                    enabled = set_enable_response.resp_state
                    if enabled == val:
                        success = True
                except Exception as e:
                    nepi_msg.printMsgWarn("ai_darknet_if: Failed to call enable request for model " + model_name + " " + str(e))
        return success

        
   

if __name__ == '__main__':
    node_name = "ai_darknet_test"
    while nepi_ros.check_for_node(node_name):
        nepi_msg.printMsgInfo("ai_darknet_if: Trying to kill running node: " + node_name)
        nepi_ros.kill_node(node_name)
        nepi_ros.sleep(2,10)
    DarknetAIF(TEST_AI_DICT,TEST_LAUNCH_NAMESPACE,TEST_MGR_NAMESPACE,TEST_MODELS_LIB_PATH)
