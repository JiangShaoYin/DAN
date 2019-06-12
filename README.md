# Deep Alignment Network #
This is a reference implementation of the face alignment method described in "Deep Alignment Network: A convolutional neural network for robust face alignment" which has been accepted to the First Faces in-the-wild Workshop-Challenge at CVPR 2017. You can read the entire paper on Arxiv [here](https://arxiv.org/abs/1706.01789). You can download the presentation and poster from Dropbox [here](https://www.dropbox.com/sh/u4g2o5kha0mt1uc/AADDMkoMKG2t4iiTxMUC6e2Ta?dl=0) or Google drive [here](https://drive.google.com/drive/folders/1QFZk_ED_FLW0xZC_gNuAKsYjLyKRMPPy).

<img src="http://home.elka.pw.edu.pl/~mkowals6/lib/exe/fetch.php?media=wiki:dan-poster.jpg" width="60%">

## Getting started ##
First of all you need to make sure you have installed Python 2.7. For that purpose we recommend Anaconda, it has all the necessary libraries except:
 * OpenCV 3.1.0 or newer
 * Lasagne 0.2（用conda安装Lasagne时，会自动安装Theano 0.9.0）
 * Theano 0.9.0

## 文件执行顺序 ##
 * CameraDemo.py 利用摄像头动态检测人脸68个特征点
 * TestSetPreparation.py 准备模型训练所需的数据
 
Once you have installed Python and the dependencies download at least one of the two pre-trained models available on Dropbox [here](https://www.dropbox.com/sh/v754z1egib0hamh/AADGX1SE9GCj4h3eDazsc0bXa?dl=0) or Google drive [here](https://drive.google.com/open?id=168tC2OxS5DjyaiuDy_JhIV3eje8K_PLJ).

The easiest way to see our method in action is to run the **CameraDemo.py** script which performs face tracking on a local webcam.

## Running the experiments from the article ##
Before continuing download the model files as described above.

### Comparison with state-of-the-art ###
Download the 300W, LFPW, HELEN, AFW and IBUG datasets from https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/ and extract them to /data/images/ into separate directories: 300W, lfpw, helen, afw and ibug.
Run the
**TestSetPreparation.py**
 it may take a while.

Use the **DANtesting.py** to perform the experiments. It will calculate the average error for all of the test subsets as well as the AUC@0.08 score and failure rate for the 300W public and private test sets.

The parameters you can set in the script are as follows:
 * verbose: if True the script will display the error for each image,
 * showResults: if True it will show the localized landmarks for each image,
 * showCED: if True the Cumulative Error Distribution curve will be shown along with the AUC score,
 * normalization: 'centers' for inter-pupil distance, 'corners' for inter-ocular distance, 'diagonal' for bounding box diagonal normalization.
 * failureThreshold: the error threshold over which the results are considered to be failures, for inter-ocular distance it should be set to 0.08,
 * networkFilename: either '../DAN.npz' or '../DAN-Menpo.npz'.


