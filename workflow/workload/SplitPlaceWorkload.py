from .Workload import *
from datetime import datetime
from workflow.database.Database import *
from random import gauss, choices
import random
import shutil  

import torch
from torchvision import datasets, transforms
import os
import bz2
import pickle
import _pickle as cPickle

class SPW(Workload):
    def __init__(self, num_workflows, std_dev, database):
        super().__init__()
        self.num_workflows = num_workflows
        self.std_dev = std_dev
        self.db = database
        self.formDatasets()
        if os.path.exists('tmp/'): shutil.rmtree('tmp/')
        os.mkdir('tmp/')

    def formDatasets(self):
        self.datasets = {}
        torch.manual_seed(1)
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
        for data_type in ['MNIST', 'FashionMNIST', 'CIFAR100']:
            dataset = eval("datasets."+data_type+"('workflow/workload/DockerImages/data', train=True, download=True,transform=transform)")
            train_loader = torch.utils.data.DataLoader(dataset, batch_size=20000, shuffle=True)
            self.datasets[data_type] = list(train_loader)
        
    def createWorkflowInput(self, data_type, workflow_id):
        path = 'tmp/'+str(workflow_id)+'/'
        if not os.path.exists(path): os.mkdir(path)
        data, target = random.choice(self.datasets[data_type])
        with bz2.BZ2File(path+str(workflow_id)+'_data.pt', 'wb') as f:
            cPickle.dump(data, f)
        with bz2.BZ2File(path+'target.pt', 'wb') as f:
            cPickle.dump(target, f)

    def generateNewWorkflows(self, interval):
        workflowlist = []
        workflows = ['MNIST', 'FashionMNIST', 'CIFAR100']
        min_sla, layer_intervals = 2, [5, 8, 15]
        max_sla = [i + (i - min_sla) for i in layer_intervals]
        max_sla_dict = dict(zip(workflows, max_sla))
        minimum_workflows = 1 if interval == 0 else 0
        for i in range(max(minimum_workflows,int(gauss(self.num_workflows, self.std_dev)))):
            WorkflowID = self.workflow_id
            workflow = random.choices(workflows, weights=[0.5, 0.25, 0.25])[0]
            SLA = np.random.randint(2,max_sla_dict[workflow])
            workflowlist.append((WorkflowID, interval, SLA, workflow))
            self.createWorkflowInput(workflow, WorkflowID)
            self.workflow_id += 1
        return workflowlist

    def generateNewContainers(self, interval, workflowlist, workflowDecision):
        workloadlist = []
        containers = []
        for i, (WorkflowID, interval, SLA, workflow) in enumerate(workflowlist):
            decision = workflowDecision[i]
            for split in range(4 if 'layer' in decision else 5):
                CreationID = self.creation_id
                application = 'shreshthtuli/'+workflow.lower()+'_'+decision
                dependentOn = CreationID - 1 if ('layer' in decision and split > 0) else None 
                workloadlist.append((WorkflowID, CreationID, interval, split, dependentOn, SLA, application))
                self.creation_id += 1
        self.createdContainers += workloadlist
        self.deployedContainers += [False] * len(workloadlist)
        return self.getUndeployedContainers()