import cv2
import numpy as np

import SkyRegionDetector
from PredictionData import EvalMetrics
import ImgUtils


# ----------------------------------------------------------

# Modifiable parameters for testing
IMG_PER_DATASET = 10
SHOW_RESULTS = True
RANDOM_SEED = 333

# ----------------------------------------------------------

# Load ground truth masks
mask623_gt = (cv2.imread("Skyfinder Dataset/623-mask.png", cv2.IMREAD_GRAYSCALE)/255).astype(np.uint8)
mask684_gt = (cv2.imread("Skyfinder Dataset/684-mask.png", cv2.IMREAD_GRAYSCALE)/255).astype(np.uint8)
mask9730_gt = (cv2.imread("Skyfinder Dataset/9730-mask.png", cv2.IMREAD_GRAYSCALE)/255).astype(np.uint8)
mask10917_gt = (cv2.imread("Skyfinder Dataset/10917-mask.png", cv2.IMREAD_GRAYSCALE)/255).astype(np.uint8)

# Load lists of random images from each dataset
dataset623 = ImgUtils.loadRandomImages("Skyfinder Dataset/623", IMG_PER_DATASET, RANDOM_SEED)
dataset684 = ImgUtils.loadRandomImages("Skyfinder Dataset/684", IMG_PER_DATASET, RANDOM_SEED)
dataset9730 = ImgUtils.loadRandomImages("Skyfinder Dataset/9730", IMG_PER_DATASET, RANDOM_SEED)
dataset10917 = ImgUtils.loadRandomImages("Skyfinder Dataset/10917", IMG_PER_DATASET, RANDOM_SEED)

results623 = []
results684 = []
results9730 = []
results10917 = []

# ----------------------------------------------------------

datasetNums = [623, 684, 9730, 10917]
datasets = [dataset623, dataset684, dataset9730, dataset10917]
gtMasks = [mask623_gt, mask684_gt, mask9730_gt, mask10917_gt]
results = [results623, results684, results9730, results10917]

for i in range(IMG_PER_DATASET):
    for d in range(len(datasets)):
        img = datasets[d][i]
        predResults, runtime = SkyRegionDetector.process(img)
        evaluate = EvalMetrics.calculate(predResults.predMask, gtMasks[d])
        evaluate.runtime = runtime
        results[d].append(evaluate)
        if (SHOW_RESULTS):
            print(predResults.timeOfDay)
            print("No sky detected!") if predResults.noSky else None
            print(evaluate)
            SkyRegionDetector.visualize(img, 
                                        predResults.predImg, 
                                        predResults.skylineCoords, 
                                        predResults.timeOfDay)
    
    print(f"{(i+1)*len(datasets)} images processed")
    
for d in range(len(datasets)):
    finalEval = EvalMetrics.aggregate(results[d])
    print(f"DATASET {datasetNums[d]}")
    print(finalEval)
    
allResults = [r for rlist in results for r in rlist]
overall = EvalMetrics.aggregate(allResults)
print("OVERALL PERFORMANCE")
print(overall)