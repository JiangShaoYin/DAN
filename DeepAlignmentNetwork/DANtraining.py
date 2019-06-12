from ImageServer import ImageServer
from FaceAlignmentTraining import FaceAlignmentTraining

datasetDir = "../data/"

# trainSet = ImageServer.Load(datasetDir + "dataset_nimgs=60960_perturbations=[0.2, 0.2, 20, 0.25]_size=[112, 112].npz") # 训练集数据
# validationSet = ImageServer.Load(datasetDir + "dataset_nimgs=100_perturbations=[]_size=[112, 112].npz")

trainSet = ImageServer.Load(datasetDir + "dataset_nimgs=14220_perturbations=[0.2, 0.2, 20, 0.25]_size=[112, 112].npz") # 训练集数据
validationSet = ImageServer.Load(datasetDir + "dataset_nimgs=100_perturbations=[]_size=[112, 112].npz")


#The parameters to the FaceAlignmentTraining constructor are: number of stages and indices of stages that will be trained
#first stage training only
training = FaceAlignmentTraining(1, [0]) # 创建人脸对齐训练对象
#second stage training only
#training = FaceAlignmentTraining(2, [1])

training.loadData(trainSet, validationSet)
training.initializeNetwork()

#load previously saved moved. training.loadNetwork("../DAN-Menpo.npz")

training.train(0.001, num_epochs=1000)