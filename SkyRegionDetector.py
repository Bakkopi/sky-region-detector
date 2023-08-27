import cv2
import numpy as np
from matplotlib import pyplot as pt
import datetime

import DayNightClassifier
from PredictionData import PredResults


# -------------------------------------------------------------

# Extract sky region information from the input image
def skyDetect(img):
    h, w, _ = img.shape
    
    # Day/night scene classification
    timeOfDay = DayNightClassifier.classifyDN(img)
    
    # Get optimum single channel image
    if (timeOfDay == "Day"):
        singleChannel = img[:, :, 2]  # Get Blue channel
    elif (timeOfDay == "Night"): 
        singleChannel = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Preprocess - Filter noise, weaken cloud edges
    singleChannel = cv2.blur(singleChannel, (9, 3))
    cv2.medianBlur(singleChannel, 5)
    
    # Canny edge detection - extract skyline borders
    edges = cv2.Canny(image=singleChannel, threshold1=50, threshold2=150)
    edgeInv = (edges < 5).astype(np.uint8)    # Inverted binary (edges = 0)
    predMask, skylinePoints = edgesToMask(edgeInv)

    # Filter no-sky images - average skyline position too close to top
    noSky = True if (np.mean(skylinePoints) < h*0.1) else False
    if (noSky):
        skylinePoints = [0]*w
        predMask = np.zeros((h, w), dtype=np.uint8)
    
    # Convert points to coordinates
    skylineCoords = [(row, col) for row, col in enumerate(skylinePoints)]

    # Extract sky region using mask
    extractedSky = cv2.bitwise_and(img, img, mask=predMask)
    predResults = PredResults(extractedSky, predMask, skylineCoords, timeOfDay, noSky)
    
    # TEST: Display image if no sky detected
    # visualize(img[:,:,::-1], extractedSky, skylineCoords) if (noSky) else None  
    
    return predResults


# Convert edge image to mask and skyline points
def edgesToMask(edgeMask):
    h, w = edgeMask.shape

    # ------ Get skyline border points (column-by-column) ------
    skylinePoints = []
    for i in range(w):
        raw = edgeMask[:, i]
        try:
            # Row index of first edge from the top
            firstEdgeIndex = np.where(raw == 0)[0][0]
            
            # Store valid border point, exclude watermark regions
            if firstEdgeIndex > 20 and firstEdgeIndex < h-20:
                skylinePoints.append(firstEdgeIndex)
            else:
                skylinePoints.append(-1)  # No valid border point
        except:
            skylinePoints.append(-1)  # No border point in column
            continue
        
    # -------- Refine sky border (impute missing points) --------
    skylinePoints = np.array(skylinePoints)
    
    # Column indexes of valid border points
    validCols = np.where(skylinePoints >= 0)[0]  
    
    if (len(validCols) > 0):
        startCol = validCols[0]  # First valid border point
        endCol = validCols[-1]   # Last valid border point
        
        # Impute with row index of adjacent points
        for i in range(startCol, endCol+1):
            if skylinePoints[i] == -1:
                skylinePoints[i] = skylinePoints[i-1]
        skylinePoints[:startCol]=skylinePoints[startCol]
        skylinePoints[endCol:]=skylinePoints[endCol]
    else:
        skylinePoints = np.array([0]*w)  # No edges detected whatsoever
        
    # -------- Create mask using final sky border points --------
    finalMask = np.ones((h, w), dtype=np.uint8)
    for i in range(w):
        finalMask[skylinePoints[i]:, i] = 0
        
    return finalMask, skylinePoints


# Perform prediction, measure processing time
def process(img):
    startTime = datetime.datetime.now()
    
    img = img[:,:,::-1]  # Reverse channel order - convert BGR to RGB
    predResults = skyDetect(img)
    
    endTime = datetime.datetime.now()
    runtime = (endTime - startTime).total_seconds() * 1000
    
    return predResults, runtime


# -------------------------------------------------------------


# Display - original, predicted, skyline images
def visualize(oriImg, predImg, skylineCoords, timeOfDay=None):
    pt.figure(figsize=(13, 7), tight_layout=True)
    # manager = pt.get_current_fig_manager()
    # manager.window.showMaximized()
    pt.subplot(1, 3, 1)
    pt.imshow(oriImg[:,:,::-1], cmap="gray")
    oriTitle = "Original" if (timeOfDay==None) else f"{timeOfDay} Image"
    pt.title(oriTitle, fontweight='bold')
    pt.axis('off')
    pt.subplot(1, 3, 2)
    pt.imshow(predImg, cmap="gray")
    pt.title("Sky Prediction", fontweight='bold')
    pt.axis('off')
    pt.subplot(1, 3, 3)
    pt.imshow(drawSkyline(oriImg, skylineCoords), cmap="gray")
    pt.title("Skyline", fontweight='bold')
    pt.axis('off')
    pt.show()
    pt.waitforbuttonpress(0) # Wait for button press
    pt.close()


# Visualize detected skyline using given coordinates
def drawSkyline(oriImg, skylineCoords):
    # Grayscale image for better visibility of drawn skyline
    grayscale = cv2.cvtColor(oriImg, cv2.COLOR_RGB2GRAY)
    newImg = cv2.merge([grayscale, grayscale, grayscale])
    
    # Convert to OpenCV contour - NumPy array (N, 1, 2)
    contour = np.array(skylineCoords).reshape(-1, 1, 2) 
    cv2.polylines(newImg, [contour], isClosed=False, color=(0,255,0), thickness=4)
    return newImg