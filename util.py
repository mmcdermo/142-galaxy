import cv2
import os
from math import sqrt
import numpy as np

imageRoot  ='images_training_rev1/'
dataFile = 'training_solutions_rev1.csv'

# Load CSV data into a hashmap keyed on the first value
def loadData(dataFile=dataFile):
    data = np.genfromtxt(dataFile, delimiter=',')
    dataHM = {}
    for i in xrange(1, len(data)):  dataHM[ int(data[i][0]) ] = data[i][1:]
    return dataHM

def loadImages(ratio, imageRoot = imageRoot):
    images = os.listdir(imageRoot)
    n = int(len(images) * ratio)
    print "Loading " + str(n) + " Images"
    loaded = []
    for i in xrange(0, n):
        if i % 1000 == 0: print "."
        loaded.append({'ID': int(images[i].split(".")[0])
                       ,"img": cv2.imread(imageRoot + images[i]) })
    return loaded

# Sequentially apply transformations to the images        
def preprocess(images, filters):
    print "\nPreprocessing images"
    processed = []
    for i in xrange(0, len(images)):
        if i % 1000 == 0: print "."
        image = images[i]
        for j in xrange(0, len(filters)): 
            image['img'] = filters[j](image['img'])
        processed.append(image)
    return processed

def loadPreprocess(ratio, filters, imageRoot=imageRoot):
    images = os.listdir(imageRoot)
    n = int(len(images) * ratio)
    print "Loading " + str(n) + " Images"
    loaded = []
    for i in xrange(0, n):
        if i % 1000 == 0: print "."
        img = cv2.imread(imageRoot + images[i])
        for j in xrange(0, len(filters)):img = filters[j](img)
        loaded.append({'ID': int(images[i].split(".")[0])
                       ,"img": img })
    return loaded
    

# Some convenience filters
def crop((x1, x2), (y1, y2)): return (lambda img: img[x1:x2, y1:y2])
def resize(x, y): return (lambda img: cv2.resize(img, (x, y)))
def grayscale(img): return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Convert CSV data and image data to 1D data and targets
def combineData(dataHM, processed):
    X = []
    Y = []
    for i in xrange(0, len(processed)): 
        x = processed[i]['img']
        X.append( x.reshape(x.shape[0] * x.shape[1]) )
        Y.append(dataHM[processed[i]['ID']])
    return (X, Y)
    
# Do everything above
def everything(ratioImages, filters, dataFile = dataFile, imageRoot = imageRoot):
    dataHM = loadData(dataFile)
    processed = loadPreprocess(ratioImages, filters, imageRoot)
    return combineData(dataHM, processed)

# Preview nx ** 2 1D arrays as images
def preview(images, x, y, nx):
    ny = nx
    out = np.zeros((x * nx, y * ny))
    out = cv2.resize(images[0].reshape(x, y), (x*nx, y*ny))
    for i in xrange(0, nx):
        for j in xrange(0, ny):
            img = images[i*nx + j].reshape(x, y)
            out[i*x:(i+1)*x, j*y:(j+1)*y] = img
    cv2.imshow("Sample", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def submission(predict, filters):
    testData = loadPreprocess(1.0, filters, "images_test_rev1/")
    testtest = []
    for i in range(len(testData)): testtest.append(testData[i]['img'])
    preview(testtest, 36, 36, 20)
    processed = []
    for i in range(len(testData)):
        x = testData[i]['img']
        processed.append( x.reshape(x.shape[0] * x.shape[1]) )
    outputs = []
    for i in range(len(processed)):
        outputs.append(predict(processed[i]))
    f = open("submission.csv", "wb")
    for i in range(len(outputs)):
        s = str(testData[i]['ID']) + ',' + ','.join([str(e) for e in np.hstack(outputs[i])])
        f.write(s+'\n')
    f.close()

def rmse(X_test, y_test, predict):
    errs = []
    for i in range(0, len(X_test)):
        if i % 100 == 0: print ","
        activ = predict(X_test[i])
        diff = y_test[i] - activ
        diffSq = [e**2 for e in diff]
        errs.append(np.sum(diffSq))
    errSum = np.sum(np.array(errs))
    return sqrt( errSum / (len(X_test) * len(X_test[0])))
