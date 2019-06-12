#	  coding:utf-8
from ImageServer import ImageServer

import numpy as np
# 训练前准备数据
# imageDirs = ["../data/images/lfpw/trainset/", "../data/images/helen/trainset/", "../data/images/afw/"] # 图像文件夹的位置
# boundingBoxFiles = ["../data/boxesLFPWTrain.pkl", "../data/boxesHelenTrain.pkl", "../data/boxesAFW.pkl"] # pkl文件保存label信息，用Pickle模块进行读写

imageDirs = ["../data/images/lfpw/trainset/"] # 图像文件夹的位置
boundingBoxFiles = ["../data/boxesLFPWTrain.pkl"] # pkl文件保存label信息，用Pickle模块进行读写

datasetDir = "../data/"

meanShape = np.load("../data/meanFaceShape.npz")["meanShape"] # 所有人脸的68个点的平均值，做为初始值，以此为基础，算偏差。

trainSet = ImageServer(initialization='rect')
trainSet.PrepareData(imageDirs, None, meanShape, 100, 100000, True)
trainSet.LoadImages()
trainSet.GeneratePerturbations(10, [0.2, 0.2, 20, 0.25]) #
trainSet.NormalizeImages()
trainSet.Save(datasetDir)

validationSet = ImageServer(initialization='box')
validationSet.PrepareData(imageDirs, boundingBoxFiles, meanShape, 0, 100, False)
validationSet.LoadImages()
validationSet.CropResizeRotateAll()
validationSet.imgs = validationSet.imgs.astype(np.float32)
validationSet.NormalizeImages(trainSet)
validationSet.Save(datasetDir)