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
pxTs = []
pyTs = []

count = 0
for i in range(np.size(trImages,0)):
    for j in range(np.size(tgImages,0)):
        tempGx = trImages[i,:,:,:]
        tempGy = trLabel[i]
        tempPx = tgImages[j,:,:,:]
        tempPy = tgLabel[j]
        if trLabel[i] == tgLabel[j]:
            print count
            count = count + 1
            gxTr.append(tempGx)
            temp = np.zeros([51], dtype='int32')
            temp[tempGy - 1] = 1
            tempGy = temp
            gyTr.append(tempGy)
            temp = np.zeros([51], dtype='int32')
            temp[tempPy-1] = 1
            tempPy = temp
            pxTr.append(tempPx)
gxTrain = np.asarray(gxTr)
gyTrain = np.asarray(gyTr)
pxTrain = np.asarray(pxTr)


np.save('ae_gxTrain',gxTrain)
np.save('ae_gyTrain',gyTrain)
np.save('ae_pxTrain',pxTrain)

