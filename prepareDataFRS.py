import os
import numpy as np
import Image
from scipy import misc
import glob


read = lambda imname: np.asarray(Image.open(imname).convert("RGB"))

dsDir = '/media/dell/VPLAB_SAMIK/IITM/'
trDir = dsDir + 'inputgallery/'
tsDir = dsDir + 'inputprobe/'

r = []
rLabel = []
t = []
tLabel = []
tg = []
tg1Label = []
count = 1

for directories in os.listdir(trDir):
    dir = os.path.join(trDir,directories)
    dir = os.path.join(dir,'results/')
    for files in os.listdir(dir):
        filepath = os.path.join(dir,files)
        print filepath
        img = misc.imread(filepath)
        img = misc.imresize(img, [100,100,3],'bicubic')
        r.append(img)
        rLabel.append(count)
        trImages = np.asarray(r)
        trLabel = np.asarray(rLabel)
    count = count+1
np.save('trainImagesIITM',trImages)
np.save('trainLabelIITM',trLabel)

count = 1

for directories in os.listdir(tsDir):
    dir = os.path.join(tsDir,directories)
    dir = os.path.join(dir,'results/')
    counter = 0
    for files in os.listdir(dir):
        filepath = os.path.join(dir,files)
        print filepath
        img = misc.imread(filepath)
        img = misc.imresize(img, [100,100,3],'bicubic')
        if counter < 5:
            tg.append(img)
            tg1Label.append(count)
        t.append(img)
        tLabel.append(count)
        tsImages = np.asarray(t)
        tsLabel = np.asarray(tLabel)
        tgImages = np.asarray(tg)
        tgLabel = np.asarray(tg1Label)
        counter = counter+1
    count = count+1
np.save('testImagesIITM',tsImages)
np.save('testLabelIITM',tsLabel)
np.save('targetImagesIITM',tgImages)
np.save('targetLabelIITM',tgLabel)