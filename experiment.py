#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 14:19:12 2025

@author: wouya
"""
import os
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = ""


from Framework import AdversarialRegulariserKeras
from networks import ConvNetClassifier,ConvNetReluClassifier, SimpleUNetClassifier, MSGAPUnetClassifier
from data_pips import ellipses, BSDS
from forward_models import CT, Denoising, Convolution
from util import gaussian_kernel

#DATA_PATH = '/local/scratch/public/sl767/LUNA/'
SAVES_PATH = 'Tests/'

KERNEL = gaussian_kernel(size=5, sigma=1)
ALPHA = 1e-2

class BSDSExperimentDeblurring(AdversarialRegulariserKeras):
    experiment_name = 'BSDS_Deconv_n0.005_5X5_sig1'
    noise_level = 0.005

    # relation between L2 error and regulariser
    # 0 corresponds to pure L2 loss, infty to pure adversarial loss
    mu_default = 0.63

    learning_rate = 0.00007
    batch_size = 32
    step_size = .00001
    lmb = 10000
    total_steps_default = 100
    starting_point = 'Mini'
    update_mini_steps = 5
    tv_coeff = 0

    def get_network(self, size, colors):
        return MSGAPUnetClassifier(size=size, colors=colors)

    def unreg_mini(self, y, fbp):
        return self.update_pic(5, y, fbp, 0, tv_coeff = self.tv_coeff)

    def get_Data_pip(self, data_path):
        return BSDS(data_path)

    def get_model(self, size):
        return Convolution(size=size, alpha=ALPHA, kernel = KERNEL)

class Experiment1(AdversarialRegulariserKeras):
    experiment_name = 'MediumNoiseV2'
    noise_level = 0.01

    # relation between L2 error and regulariser
    # 0 corresponds to pure L2 loss, infty to pure adversarial loss
    mu_default = 2.56 

    learning_rate = 0.0001
    batch_size = 256
    step_size = .00001
    lmb = 100
    total_steps_default = 10
    starting_point = 'Mini'

    def get_network(self, size, colors):
        return ConvNetClassifier(size=size, colors=colors)

    def unreg_mini(self, y, fbp):
        return self.update_pic(10, y, fbp, 0, tv_coeff = self.tv_coeff)

    def get_Data_pip(self, data_path):
        return ellipses(data_path)

    def get_model(self, size):
        return Denoising(size=size)


class BSDSExperiment1(AdversarialRegulariserKeras):
    experiment_name = 'BSDS_0.05_no_TV'
    noise_level = 0.05

    # relation between L2 error and regulariser
    # 0 corresponds to pure L2 loss, infty to pure adversarial loss
    mu_default = 22.16

    learning_rate = 0.00007
    batch_size = 256
    step_size = .00001
    lmb = 10000
    total_steps_default = 50
    starting_point = 'Mini'
    update_mini_steps = 10
    tv_coeff =0

    def get_network(self, size, colors):
        return MSGAPUnetClassifier(size=size, colors=colors)

    def unreg_mini(self, y, fbp):
        return self.update_pic(10, y, fbp, 0, tv_coeff = self.tv_coeff)

    def get_Data_pip(self, data_path):
        return BSDS(data_path)

    def get_model(self, size):
        return Denoising(size=size)

class BSDSExperiment2(AdversarialRegulariserKeras):
    experiment_name = 'BSDS_0.05_TV_0.01'
    noise_level = 0.05

    # relation between L2 error and regulariser
    # 0 corresponds to pure L2 loss, infty to pure adversarial loss
    mu_default = 22.16

    learning_rate = 0.00007
    batch_size = 256
    step_size = .00001
    lmb = 10000
    total_steps_default = 50
    starting_point = 'Mini'
    update_mini_steps = 1
    tv_coeff = 10e-2

    def get_network(self, size, colors):
        return MSGAPUnetClassifier(size=size, colors=colors)

    def unreg_mini(self, y, fbp):
        return self.update_pic(1, y, fbp, 0, tv_coeff = self.tv_coeff)

    def get_Data_pip(self, data_path):
        return BSDS(data_path)

    def get_model(self, size):
        return Denoising(size=size)


class BSDSExperiment3(AdversarialRegulariserKeras):
    experiment_name = 'BSDS_0.05_TV_no'
    noise_level = 0.05

    # relation between L2 error and regulariser
    # 0 corresponds to pure L2 loss, infty to pure adversarial loss
    mu_default = 22.16

    learning_rate = 0.00007
    batch_size = 512
    step_size = .00001
    lmb = 10000
    total_steps_default = 30
    starting_point = 'Mini'
    update_mini_steps = 10
    tv_coeff = 0

    def get_network(self, size, colors):
        return MSGAPUnetClassifier(size=size, colors=colors)

    def unreg_mini(self, y, fbp):
        return self.update_pic(10, y, fbp, 0, tv_coeff = self.tv_coeff)

    def get_Data_pip(self, data_path):
        return BSDS(data_path)

    def get_model(self, size):
        return Denoising(size=size)


class BSDSExperiment4(AdversarialRegulariserKeras):
    experiment_name = 'BSDS_0.1_TV_no'
    noise_level = 0.1

    # relation between L2 error and regulariser
    # 0 corresponds to pure L2 loss, infty to pure adversarial loss
    mu_default = 44.3

    learning_rate = 0.00007
    batch_size = 512
    step_size = .00001
    lmb = 10000
    total_steps_default = 10
    starting_point = 'Mini'
    update_mini_steps = 10
    tv_coeff = 0

    def get_network(self, size, colors):
        return MSGAPUnetClassifier(size=size, colors=colors)

    def unreg_mini(self, y, fbp):
        return self.update_pic(5, y, fbp, 0, tv_coeff = self.tv_coeff)

    def get_Data_pip(self, data_path):
        return BSDS(data_path)

    def get_model(self, size):
        return Denoising(size=size)

class Experiment2(AdversarialRegulariserKeras):
    experiment_name = 'Moderate+SimpleUnet'
    noise_level = 0.1

    # relation between L2 error and regulariser
    # 0 corresponds to pure L2 loss, infty to pure adversarial loss
    mu_default = 25.58
    lmb = 1000
    learning_rate = 0.00007
    step_size = 1
    total_steps_default = 500
    starting_point = 'Mini'
    update_mini_steps = 10
    tv_coeff = 0

    def get_network(self, size, colors):
        return SimpleUNetClassifier(size=size, colors=colors)

    def unreg_mini(self, y, fbp):
        return self.update_pic(10, y, fbp, 0, tv_coeff = self.tv_coeff)


    def get_Data_pip(self, data_path):
        return ellipses(data_path)

    def get_model(self, size):
        return Denoising(size=size)
        

class Experiment2_TV(AdversarialRegulariserKeras):
    experiment_name = 'Moderate+SimpleUnet_TV'
    noise_level = 0.1

    # relation between L2 error and regulariser
    # 0 corresponds to pure L2 loss, infty to pure adversarial loss
    mu_default = 25.58
    lmb = 1000
    learning_rate = 0.00007
    step_size = 1
    total_steps_default = 50
    starting_point = 'Mini'
    update_mini_steps = 5
    tv_coeff = 10e-3

    def get_network(self, size, colors):
        return SimpleUNetClassifier(size=size, colors=colors)

    def unreg_mini(self, y, fbp):
        return self.update_pic(5, y, fbp, 0, tv_coeff = self.tv_coeff)


    def get_Data_pip(self, data_path):
        return ellipses(data_path)

    def get_model(self, size):
        return Denoising(size=size)
        


class Experiment3_TV(AdversarialRegulariserKeras):
    experiment_name = 'Moderate+MSGAPUnet+TV'
    noise_level = 0.1

    # relation between L2 error and regulariser
    # 0 corresponds to pure L2 loss, infty to pure adversarial loss
    mu_default = 25.58
    lmb = 1000
    learning_rate = 0.00007
    step_size = .1
    total_steps_default = 50
    starting_point = 'Mini'
    update_mini_steps = 2
    tv_coeff = 10e-2

    def get_network(self, size, colors):
        return MSGAPUnetClassifier(size=size, colors=colors)

    def unreg_mini(self, y, fbp):
        return self.update_pic(2, y, fbp, 0, tv_coeff = self.tv_coeff)

    def get_Data_pip(self, data_path):
        return ellipses(data_path)

    def get_model(self, size):
        return Denoising(size=size)


class Experiment4_TV(AdversarialRegulariserKeras):
    experiment_name = 'Medium+MSGAPUnet+TV'
    noise_level = 0.05

    # relation between L2 error and regulariser
    # 0 corresponds to pure L2 loss, infty to pure adversarial loss
    mu_default = 12.8
    lmb = 1000
    learning_rate = 0.00007
    step_size = .1
    total_steps_default = 50
    starting_point = 'Mini'
    update_mini_steps = 1
    tv_coeff = 10e-2

    def get_network(self, size, colors):
        return MSGAPUnetClassifier(size=size, colors=colors)

    def unreg_mini(self, y, fbp):
        return self.update_pic(1, y, fbp, 0, tv_coeff = self.tv_coeff)

    def get_Data_pip(self, data_path):
        return ellipses(data_path)

    def get_model(self, size):
        return Denoising(size=size)



class Experiment4(AdversarialRegulariserKeras):
    experiment_name = 'MediumNoise0.02_stepsize=.005'
    noise_level = 0.02

    # relation between L2 error and regulariser
    # 0 corresponds to pure L2 loss, infty to pure adversarial loss
    mu_default = 5.12026

    learning_rate = 0.0001
    lmb = 100
    step_size = .7
    total_steps_default = 50
    starting_point = 'Mini'

    def get_network(self, size, colors):
        return ConvNetClassifier(size=size, colors=colors)

    def unreg_mini(self, y, fbp):
        return self.update_pic(10, y, fbp, 0)

    def get_Data_pip(self, data_path):
        return ellipses(data_path)

    def get_model(self, size):
        return Denoising(size=size)
    
    



def main():
    experiment = Experiment1("", saves_path=SAVES_PATH)
    experiment.find_good_lambda(64)
    for k in range(100):
        experiment.train(100)

def main2():
    experiment = Experiment2("", saves_path=SAVES_PATH)
    experiment.find_good_lambda(64)
    for k in range(100):
        experiment.train(100)

def moderate_simpleUnet():
    experiment = Experiment2("", saves_path=SAVES_PATH)
    experiment.find_good_lambda(64)
    for k in range(100):
        experiment.train(100)


def moderate_simpleUnet_TV():
    experiment = Experiment2_TV("", saves_path=SAVES_PATH)
    experiment.find_good_lambda(64)
    for k in range(100):
        experiment.train(100)

def moderate_simpleUnet_TV():
    experiment = Experiment2_TV("", saves_path=SAVES_PATH)
    experiment.find_good_lambda(64)
    for k in range(100):
        experiment.train(100)

def moderate_MSGAPUnet_TV():
    experiment = Experiment3_TV("", saves_path=SAVES_PATH)
    experiment.find_good_lambda(64)
    for k in range(10):
        experiment.train(100)
def medium_MSGAPUnet_TV():
    experiment = Experiment4_TV("", saves_path=SAVES_PATH)
    experiment.find_good_lambda(64)
    for k in range(10):
        experiment.train(100)


def BSDS_experiment1():
    experiment = BSDSExperiment1("BSDS300/", saves_path=SAVES_PATH)
    experiment.find_good_lambda(64)
    for k in range(10):
        experiment.train(100)



def BSDS_experiment2():
    experiment = BSDSExperiment2("BSDS300/", saves_path=SAVES_PATH)
    experiment.find_good_lambda(64)
    for k in range(10):
        experiment.train(100)

def BSDS_experiment3():
    experiment = BSDSExperiment3("BSDS300/", saves_path=SAVES_PATH)
    experiment.find_good_lambda(64)
    for k in range(10):
        experiment.train(100)

def BSDS_experiment4():
    experiment = BSDSExperiment4("BSDS300/", saves_path=SAVES_PATH)
    experiment.find_good_lambda(64)
    for k in range(10):
        experiment.train(100)
        
def BSDS_experimentDeblurring():
    experiment = BSDSExperimentDeblurring("BSDS300/", saves_path=SAVES_PATH)
    experiment.find_good_lambda(64)
    for k in range(10):
        experiment.train(100)
        
if __name__ == "__main__":
    experiment = Experiment1("BSDS300/", saves_path=SAVES_PATH)
    experiment.find_good_lambda(64)
    experiment.mu_default = 44.3
    for k in range(100):
        experiment.train(100)
    #experiment.log_optimization(16, 2000, 0.0005, None)
    # experiment.log_optimization(32, 200, 0.7, .4)
    # experiment.log_optimization(32, 200, 0.7, .5)
