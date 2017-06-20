import numpy as np


trImages = np.load('trainImagesIITM.npy')
trLabel = np.load('trainLabelIITM.npy')
tsImages = np.load('testImagesIITM.npy')
tsLabel = np.load('testLabelIITM.npy')
tgImages = np.load('targetImagesIITM.npy')
tgLabel = np.load('targetLabelIITM.npy')

gxTr = []
gyTr = []
pxTr = []
pyTr = []
vyTr = []
pxTs = []
pyTs = []


for i in range(np.size(trImages,0)):
    for j in range(np.size(tgImages,0)):
        tempGx = trImages[i,:,:,:]
        tempGy = trLabel[i]
        gxTr.append(tempGx)
        temp = np.zeros([51], dtype='int32')
        temp[tempGy-1] = 1
        tempGy = temp
        gyTr.append(tempGy)
        tempPx = tgImages[j,:,:,:]
        tempPy =tgLabel[j]
        if trLabel[i] == tgLabel[j]:
            vyTr.append([1,0])
        else:
            vyTr.append([0,1])
        temp = np.zeros([51], dtype='int32')
        temp[tempPy-1] = 1
        tempPy = temp
        pxTr.append(tempPx)
        pyTr.append(tempPy)
gxTrain = np.asarray(gxTr)
gyTrain = np.asarray(gyTr)
pxTrain = np.asarray(pxTr)
pyTrain = np.asarray(pyTr)
vyTrain = np.asarray(vyTr)

np.save('gxTrain',gxTrain)
np.save('gyTrain',gyTrain)
np.save('pxTrain',pxTrain)
np.save('pyTrain',pyTrain)
np.save('vyTrain',vyTrain)
