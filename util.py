import cv2
import os
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
        if i % 1000 == 0: print ".",
        loaded.append({'ID': int(images[i].split(".")[0])
                       ,"img": cv2.imread(imageRoot + images[i]) })
    return loaded

# Sequentially apply transformations to the images        
def preprocess(images, filters):
    print "\nPreprocessing images"
    processed = []
    for i in xrange(0, len(images)):
        if i % 1000 == 0: print ".",
        image = images[i]
        for j in xrange(0, len(filters)): 
            image['img'] = filters[j](image['img'])
        processed.append(image)
    return processed

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
def loadProcess(ratioImages, filters, dataFile = dataFile, imageRoot = imageRoot):
    dataHM = loadData(dataFile)
    loaded = loadImages(ratioImages, imageRoot)
    processed = preprocess(loaded, filters)
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
