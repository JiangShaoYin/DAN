#	  coding:utf-8

from scipy import ndimage
import numpy as np
import utils
import pickle
import glob
from os import path

class ImageServer(object):
    def __init__(self, imgSize=[112, 112], frameFraction=0.25, initialization='box', color=False):
        self.origLandmarks = []
        self.filenames = []
        self.mirrors = []
        self.meanShape = np.array([])

        self.meanImg = np.array([])
        self.stdDevImg = np.array([])

        self.perturbations = []

        self.imgSize = imgSize
        self.frameFraction = frameFraction
        self.initialization = initialization
        self.color = color;

        self.boundingBoxes = []

    @staticmethod
    def Load(filename):
        imageServer = ImageServer()
        arrays = np.load(filename)
        imageServer.__dict__.update(arrays)

        if (len(imageServer.imgs.shape) == 3):
            imageServer.imgs = imageServer.imgs[:, np.newaxis]

        return imageServer

    def Save(self, datasetDir, filename=None):
        if filename is None:
            filename = "dataset_nimgs={0}_perturbations={1}_size={2}".format(len(self.imgs), list(self.perturbations), self.imgSize)
            if self.color:
                filename += "_color={0}".format(self.color)
            filename += ".npz"

        arrays = {key:value for key, value in self.__dict__.items() if not key.startswith('__') and not callable(key)}
        np.savez(datasetDir + filename, **arrays)

    def PrepareData(self, imageDirs, boundingBoxFiles, meanShape, startIdx, nImgs, mirrorFlag):
        filenames = [] # pic文件名list
        landmarks = [] # 坐标点文件名list
        boundingBoxes = []


        for i in range(len(imageDirs)): # 训练数据来自lfpw，helen，afw这3个数据集，文件夹长度为3。
            filenamesInDir = glob.glob(imageDirs[i] + "*.jpg")
            filenamesInDir += glob.glob(imageDirs[i] + "*.png") # 打开第一个文件夹的所有jpg和png图片，文件路径 '../data/images/lfpw/trainset\\image_0020.png'

            if boundingBoxFiles is not None: #
                boundingBoxDict = pickle.load(open(boundingBoxFiles[i], 'rb'), encoding='latin1')

            for j in range(len(filenamesInDir)):
                filenames.append(filenamesInDir[j]) # 将每1个pic的路径，加入filenames所维护的list中
                
                ptsFilename = filenamesInDir[j][:-3] + "pts"
                landmarks.append(utils.loadFromPts(ptsFilename)) # 将每个图片标注信息的pts路径，写入list：'../data/images/lfpw/trainset\\image_0001.pts'

                if boundingBoxFiles is not None:
                    basename = path.basename(filenamesInDir[j])
                    boundingBoxes.append(boundingBoxDict[basename])
                

        filenames = filenames[startIdx : startIdx + nImgs]
        landmarks = landmarks[startIdx : startIdx + nImgs]
        boundingBoxes = boundingBoxes[startIdx : startIdx + nImgs]

        mirrorList = [False for i in range(nImgs)]
        if mirrorFlag:     
            mirrorList = mirrorList + [True for i in range(nImgs)]
            filenames = np.concatenate((filenames, filenames))

            landmarks = np.vstack((landmarks, landmarks)) # 6096个pic对应的特征点信息
            boundingBoxes = np.vstack((boundingBoxes, boundingBoxes))       

        self.origLandmarks = landmarks # 将读取到的信息，写到对应的mageServer的属性成员上
        self.filenames = filenames
        self.mirrors = mirrorList
        self.meanShape = meanShape
        self.boundingBoxes = boundingBoxes

    def LoadImages(self):
        self.imgs = []
        self.initLandmarks = []
        self.gtLandmarks = []

        for i in range(len(self.filenames)): # 读取文件夹下面的每个图片
            img = ndimage.imread(self.filenames[i]) # 图片shape：539 * 410 * 3

            if self.color:
                if len(img.shape) == 2:
                    img = np.dstack((img, img, img))
            else:
                if len(img.shape) > 2: # img的信息坐标轴个数，row，column，channel共3个坐标轴
                    img = np.mean(img, axis=2) # Compute the arithmetic mean along the specified axis.将rgb数据求平均，融合成新的数据
            img = img.astype(np.uint8)

            if self.mirrors[i]:
                self.origLandmarks[i] = utils.mirrorShape(self.origLandmarks[i], img.shape)
                img = np.fliplr(img)

            if self.color:
                img = np.transpose(img, (2, 0, 1))
            else:
                img = img[np.newaxis] # 在第一维增加一个新的坐标轴，shape由539 * 410，变为1 * 539 * 410

            groundTruth = self.origLandmarks[i] # 取到

            if self.initialization == 'rect':
                bestFit = utils.bestFitRect(groundTruth, self.meanShape) # 返回由平均特征点经过缩放和偏移，生成的初始特征点坐标。
            elif self.initialization == 'similarity':
                bestFit = utils.bestFit(groundTruth, self.meanShape)
            elif self.initialization == 'box':
                bestFit = utils.bestFitRect(groundTruth, self.meanShape, box=self.boundingBoxes[i])

            self.imgs.append(img) # 把文件夹中的图片依次加入list中
            self.initLandmarks.append(bestFit) # 把文件夹中的图片对应的初始landmark坐标依次加入list中
            self.gtLandmarks.append(groundTruth)
            if i % 500 == 0:
              print("the %dth pic in func LoadImages()"%i)

        self.initLandmarks = np.array(self.initLandmarks)
        self.gtLandmarks = np.array(self.gtLandmarks)    

    def GeneratePerturbations(self, nPerturbations, perturbations): # 做扰动(对输入图片进行偏移，和缩放)
        self.perturbations = perturbations
        meanShapeSize = max(self.meanShape.max(axis=0) - self.meanShape.min(axis=0))
        destShapeSize = min(self.imgSize) * (1 - 2 * self.frameFraction)
        scaledMeanShape = self.meanShape * destShapeSize / meanShapeSize

        newImgs = []  
        newGtLandmarks = []
        newInitLandmarks = []           

        translationMultX, translationMultY, rotationStdDev, scaleStdDev = perturbations

        rotationStdDevRad = rotationStdDev * np.pi / 180         
        translationStdDevX = translationMultX * (scaledMeanShape[:, 0].max() - scaledMeanShape[:, 0].min())
        translationStdDevY = translationMultY * (scaledMeanShape[:, 1].max() - scaledMeanShape[:, 1].min())
        print ("Creating perturbations of " + str(self.gtLandmarks.shape[0]) + " shapes")

        for i in range(self.initLandmarks.shape[0]):
            print(i)
            for j in range(nPerturbations):
                tempInit = self.initLandmarks[i].copy() # 将生成的S0特征点信息copy过来

                angle = np.random.normal(0, rotationStdDevRad) # 旋转角度
                offset = [np.random.normal(0, translationStdDevX), np.random.normal(0, translationStdDevY)] # 二维信息，x与y方向的偏移值
                scaling = np.random.normal(1, scaleStdDev)

                R = np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]])     # shape = 2 * 2
            
                tempInit = tempInit + offset # 将生成的landmark进行随机偏移
                tempInit = (tempInit - tempInit.mean(axis=0)) * scaling + tempInit.mean(axis=0)            
                tempInit = np.dot(R, (tempInit - tempInit.mean(axis=0)).T).T + tempInit.mean(axis=0)

                tempImg, tempInit, tempGroundTruth = self.CropResizeRotate(self.imgs[i], tempInit, self.gtLandmarks[i])                

                newImgs.append(tempImg)
                newInitLandmarks.append(tempInit)
                newGtLandmarks.append(tempGroundTruth)

        self.imgs = np.array(newImgs)
        self.initLandmarks = np.array(newInitLandmarks)
        self.gtLandmarks = np.array(newGtLandmarks)

    def CropResizeRotateAll(self):
        newImgs = []  
        newGtLandmarks = []
        newInitLandmarks = []   
        
        for i in range(self.initLandmarks.shape[0]):
            tempImg, tempInit, tempGroundTruth = self.CropResizeRotate(self.imgs[i], self.initLandmarks[i], self.gtLandmarks[i])

            newImgs.append(tempImg)
            newInitLandmarks.append(tempInit)
            newGtLandmarks.append(tempGroundTruth)

        self.imgs = np.array(newImgs)
        self.initLandmarks = np.array(newInitLandmarks)
        self.gtLandmarks = np.array(newGtLandmarks)  

    def NormalizeImages(self, imageServer=None):
        self.imgs = self.imgs.astype(np.float32)

        if imageServer is None:
            self.meanImg = np.mean(self.imgs, axis=0)
        else:
            self.meanImg = imageServer.meanImg

        self.imgs = self.imgs - self.meanImg
        
        if imageServer is None:
            self.stdDevImg = np.std(self.imgs, axis=0)
        else:
            self.stdDevImg = imageServer.stdDevImg
        
        self.imgs = self.imgs / self.stdDevImg

        from matplotlib import pyplot as plt  

        meanImg = self.meanImg - self.meanImg.min()
        meanImg = 255 * meanImg / meanImg.max()  
        meanImg = meanImg.astype(np.uint8)   
        if self.color:
            plt.imshow(np.transpose(meanImg, (1, 2, 0)))
        else:
            plt.imshow(meanImg[0], cmap=plt.cm.gray)
        plt.savefig("../meanImg.jpg")
        plt.clf()

        stdDevImg = self.stdDevImg - self.stdDevImg.min()
        stdDevImg = 255 * stdDevImg / stdDevImg.max()  
        stdDevImg = stdDevImg.astype(np.uint8)   
        if self.color:
            plt.imshow(np.transpose(stdDevImg, (1, 2, 0)))
        else:
            plt.imshow(stdDevImg[0], cmap=plt.cm.gray)
        plt.savefig("../stdDevImg.jpg")
        plt.clf()

    def CropResizeRotate(self, img, initShape, groundTruth):
        meanShapeSize = max(self.meanShape.max(axis=0) - self.meanShape.min(axis=0))
        destShapeSize = min(self.imgSize) * (1 - 2 * self.frameFraction)

        scaledMeanShape = self.meanShape * destShapeSize / meanShapeSize

        destShape = scaledMeanShape.copy() - scaledMeanShape.mean(axis=0)
        offset = np.array(self.imgSize[::-1]) / 2
        destShape += offset

        A, t = utils.bestFit(destShape, initShape, True)
    
        A2 = np.linalg.inv(A)
        t2 = np.dot(-t, A2)

        outImg = np.zeros((img.shape[0], self.imgSize[0], self.imgSize[1]), dtype=img.dtype)
        for i in range(img.shape[0]):
            outImg[i] = ndimage.interpolation.affine_transform(img[i], A2, t2[[1, 0]], output_shape=self.imgSize)

        initShape = np.dot(initShape, A) + t

        groundTruth = np.dot(groundTruth, A) + t
        return outImg, initShape, groundTruth


