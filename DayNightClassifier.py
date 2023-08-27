import cv2
import numpy as np
import ImgUtils


# Classify time of day based on mean intensity value
def classifyDN(img):
    img = img
    dimensions = img.shape
    timeOfDay = "" 
    if np.sum(np.sum(img)) / (dimensions[0] * dimensions[1] * dimensions[2]) > 97.0:
        timeOfDay = "Day"
    else:
        timeOfDay = "Night"
    return timeOfDay


# Calculate optimal mean intensity threshold for dataset
# NOTE: Images labelled with first letter of file name (d - Day / n - Night)
def getIntensityThreshold(images):
    dayMean = []
    nightMean = []
    
    for i in images:
        img = i[0]
        fileName = i[1]
        dimensions = img.shape
        if fileName.startswith("d"):
            dayMean.append(np.sum(np.sum(img)) / (dimensions[0] * dimensions[1] * dimensions[2]))
        elif fileName.startswith("n"):
            nightMean.append(np.sum(np.sum(img)) / (dimensions[0] * dimensions[1] * dimensions[2]))
        else:
            pass
    
    finalThresh = (np.mean(dayMean) + np.mean(nightMean)) / 2.0
    print("Threshold: ", finalThresh)
    return finalThresh


# Test classification accuracy given a labelled dataset 
def testAccuracy(labelledImages):
    # Parse ground truth classes for each image
    tags = []
    for i in labelledImages:
        img = i[0]
        fileName = i[1]
        if fileName.startswith("d"):
            tags.append("Day")
        elif fileName.startswith("n"):
            tags.append("Night")
    
    # Day night prediction + calculate accuracy
    total = len(labelledImages)
    correct = 0
    for i, img in enumerate(labelledImages):
        prediction = classifyDN(img[0])
        if (prediction == tags[i]):
            correct += 1
    accuracy = correct / total * 100
    print(f"Accuracy: {accuracy}")


# ----- EVALUATE ACCURACY -----
# testAccuracy(ImgUtils.loadAllImages("Day-Night Classified"))
