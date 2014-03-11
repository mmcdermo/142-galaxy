import cv2
import sys
import os
import itertools
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
        if i % 1000 == 0: print str((i/len(images)))+"%"
        image = images[i]
        for j in xrange(0, len(filters)): 
            image['img'] = filters[j](image['img'])
        processed.append(image)
    return processed

def loadPreprocess(ratio, filters, imageRoot=imageRoot, color=cv2.COLOR_BGR2GRAY, name="default"):
    cache = "cache_" + name + "_" + imageRoot
    images = os.listdir(imageRoot)
    n = int(len(images) * ratio)
    if(os.path.exists(cache)):
        cachedImages = os.listdir(cache)
        if n <= len(cachedImages):
            print "Loading images from cache "+cache
            r = []
            for i in xrange(0, n):
                if i % 10000 == 0: print str((float(i)/n*100))+"%"
                img = cv2.imread(cache + cachedImages[i])
                img = cv2.cvtColor(img, color)
                r.append({'ID': int(images[i].split(".")[0])
                          ,"img": img})
            return r
        else: print "Need to regenerate cache..."
    else: 
        print "Creating Cache: " + cache
        os.mkdir(cache)
    print "Loading " + str(n) + " New Images"
    loaded = []
    for i in xrange(0, n):
        if i % 1000 == 0: print str((float(i)/n*100))+"%"
        img = cv2.imread(imageRoot + images[i])
        if(img == None or len(img) == 0 or len(img[0]) == 0):
            print "EMPTY IMAGE"
        else:
            for j in xrange(0, len(filters)):img = filters[j](img)
            cv2.imwrite(cache + images[i], img)
            loaded.append({'ID': int(images[i].split(".")[0])
                           ,"img": img })
    return loaded
    

# Some convenience filters
def crop((x1, x2), (y1, y2)): return (lambda img: img[x1:x2, y1:y2])
def resize(x, y): return (lambda img: cv2.resize(img, (x, y)))
def grayscale(img): return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def keypoints(img): 
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#    sift = cv2.SIFT()
#    kp = sift.detect(img, None)
#    return cv2.drawKeypoints(gray, kp)
    return cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)


def laplacian(img):
    return cv2.Laplacian(img, cv2.CV_64F)

def denoise(img):
    return cv2.fastNlMeansDenoising(img,None,10,7,21)


# Convert CSV data and image data to 1D data and targets
def combineData(dataHM, processed):
    X = []
    Y = []
    for i in xrange(0, len(processed)): 
        x = processed[i]['img']
        X.append( x.reshape(x.shape[0] * x.shape[1] ) )
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

def submission(predict, filters, transform = 0):
    testData = loadPreprocess(1.0, filters, "images_test_rev1/")
    testtest = []
    for i in range(len(testData)): testtest.append(testData[i]['img'])
    #preview(testtest, 36, 36, 20)
    processed = []
    for i in range(len(testData)):
        x = testData[i]['img']
        processed.append( x.reshape(x.shape[0] * x.shape[1]) )
    if transform != 0:
        print "Transforming submission data...."
        processed = transform(processed)
        print "\t" + str(processed[0])
        print "\t" + str(processed.shape)
    outputs = []
    for i in range(len(processed)):
        outputs.append(predict(processed[i]))
    f = open("submission.csv", "wb")
    for i in range(len(outputs)):
        s = str(testData[i]['ID']) + ',' + ','.join([str(e) for e in np.hstack(outputs[i])])
        f.write(s+'\n')
    f.close()

def rmse(X_test, y_test, predict, featureErrors = False):
    errs = []
    featErrs = np.zeros(y_test[0].shape)
    for i in range(0, len(X_test)):
        #if i % 100 == 0: print ","
        activ = predict(X_test[i])
        diff = y_test[i] - activ
        diffSq = np.array([e**2 for e in diff])
        featErrs += diffSq[0]
        #print "DiffSq: " + str(diffSq)
        #oprint "FeatErrs: " + str(featErrs)
        errs.append(np.sum(diffSq))
    errSum = np.sum(np.array(errs))
    r = sqrt( errSum / (len(y_test[0]) * len(y_test)))
    
    if not featureErrors:
        return r
    else:
        return (r, np.sqrt( featErrs / len(y_test) ))

def accumulate(l):
    s = 0
    for i in l:
        s += i
        yield s
    

tasks = [3, 2, 2, 2, 4, 2, 3, 7, 3, 3, 6] # num questions per task
taskIdx = [0] + list(accumulate(tasks))   # indices of first task question

# Simplified representation of the decision tree:
taskPaths = [[]         #t1
             , [[1, 2]] #t2
             , [[2, 2]] #t3
             , [[2, 2]] #t4 = 3 = [2,2]
             , [[2, 2]] #t5 = [4,2] or 11 = [4,2] or [4,1] = [4] = [3] = [2,2]
             , []       #t6 = 5 or 9 = 3 or 9 = [[2, 2], [2, 1]] = [2] = [1,2]
                        # However, they re-normalized 6. 
             , [[1, 1]] #t7
             , [[6, 1]] #t8
             , [[2, 1]] #t9
             , [[4, 1]] #t10
             , [[4, 1]] #t11 = 10 = [4, 1]
             ]

# Undoes the galaxy zoo data weighting 
def deWeight(X):
    realProbs = np.copy(X)
    for task in range(len(taskPaths)):
        dep = taskPaths[task]
        # Fortunately all tasks rely only on one previous response
        if len(dep) > 0:
            depIdx = taskIdx[dep[0][0] - 1] + (dep[0][1] - 1)
            prob = X[depIdx]
            if prob == 0: prob = 1
            for taskFeature in range (tasks[task]):
                idx = taskIdx[task] + taskFeature
                realProbs[idx] = X[idx] / prob
    return realProbs

# This is a pseudo-softmax (no exp) that ensures responses for each question sum
#  to one before reweighting. 
def deWeightedSoftmax(X):
    Xe = np.copy(X)
    for task in range(len(taskPaths)):
        sum = 0
        for taskFeature in range (tasks[task]):
            sum += Xe[taskIdx[task] + taskFeature]
        if sum != 0:
            for taskFeature in range (tasks[task]):
                idx = taskIdx[task] + taskFeature
                Xe[idx] = Xe[idx] / sum
    return Xe

#     
def reWeight(X):
    weightedProbs = np.copy(X)
    for task in range(len(taskPaths)):
        dep = taskPaths[task]
        if len(dep) > 0:
            depIdx = taskIdx[dep[0][0] - 1] + (dep[0][1] - 1)
            prob = weightedProbs[depIdx]
            for taskFeature in range (tasks[task]):
                idx = taskIdx[task] + taskFeature
                weightedProbs[idx] = X[idx] * prob
    return weightedProbs    
    
