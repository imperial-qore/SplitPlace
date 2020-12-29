import logging
import json
import pdb
import sys
import docker
import requests
import configparser
import docker 
import codes
import json
import os

# Reference: https://docker-py.readthedocs.io/en/stable/api.html

class DockerClient():
    
    def __init__(self, dockerURL):
        self.dclient = docker.DockerClient(base_url=dockerURL)
        self.dclient1 = docker.APIClient(base_url=dockerURL)
       
    def create(self, config):
        rc = codes.SUCCESS
        name = config["name"]
        image = config["image"]
        inputFileName = config["inputFileName"]
        outputFileName = config["outputFileName"]
        try:
            cmd = "docker run -itd --name "+name+" -v ~/container_data/:/data "+image+" 'python3' 'main.py' '"+inputFileName+"' '"+outputFileName+"'"
            cid = subprocess.Popen(cmd, shell=True,stdout=subprocess.PIPE).communicate()[0].decode('utf-8').strip()
        except requests.exceptions.ConnectionError as err:
            rc = codes.FAILED
        return rc, json.dumps({"rc": rc})

    def start(self, name):
        rc = codes.SUCCESS
        try:
            self.dclient1.start(name)
        except docker.errors.NotFound as err:
            rc = codes.NOT_FOUND
        except requests.exceptions.ConnectionError as err:
            rc = codes.FAILED
        return rc, json.dumps({"rc": rc})
   
    def stop(self, name):
        rc = codes.SUCCESS
        try:
            self.dclient1.stop(container=containerId)
        except docker.errors.NotFound as err:
            rc = codes.NOT_FOUND
        except requests.exceptions.ConnectionError as err:
            rc = codes.FAILED
        return rc, json.dumps({"rc": rc})
    
    # TODO: return finish time UTC
    def delete(self, config):
        name = config["name"]
        outputFileName = config["outputFileName"]
        rc = codes.SUCCESS
        subprocess.call(["sudo","rm","-rf","/tmp/container_data/"+outputFileName])
        try:
            self.dclient1.stop(name)
            self.dclient1.remove_container(name)
        except docker.errors.NotFound as err:
            rc = codes.NOT_FOUND
        except requests.exceptions.ConnectionError as err:
            rc = codes.FAILED
        return rc, json.dumps({"rc": rc})
    
    def listContainers(self):
        containerList = []
        rc = codes.SUCCESS
        try:
            containerList = self.dclient.containers.list()
        except docker.errors.NotFound as err:
            rc = codes.NOT_FOUND
            logging.error(err) 
        except requests.exceptions.ConnectionError as e:
            rc = codes.FAILED
            logging.error(e)
        else: 
            return rc, containerList

    def inspectContainer(self, containerId):
        containerInfo = dict()
        rc = codes.SUCCESS
        try:
            containerInfo = self.dclient1.inspect_container(containerId)
        except docker.errors.NotFound as err:
            rc = codes.NOT_FOUND
            logging.error(err) 
        except requests.exceptions.ConnectionError as e:
            rc = codes.FAILED
            logging.error(e)
        return rc, json.dumps(containerInfo)
    
    def stats(self, containerId):
        rc = codes.SUCCESS
        try:
            data = self.dclient1.stats(container=containerId,decode=None,stream=False)
        except docker.errors.NotFound as err:
            rc = codes.NOT_FOUND
        except requests.exceptions.ConnectionError as err:
            rc = codes.FAILED
        return rc, data
    

                                 
