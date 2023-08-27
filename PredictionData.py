import numpy as np
from matplotlib import pyplot as pt

# -------------------------------------------------------------

# Stores segmentation and prediction results output by the algorithm
class PredResults:
    def __init__(self, predImg, predMask, skylineCoords, timeOfDay, noSky):
        self.predImg = predImg 
        self.predMask = predMask 
        self.skylineCoords = skylineCoords  
        self.timeOfDay = timeOfDay 
        self.noSky = noSky 

# -------------------------------------------------------------

# Stores and calculates performance metrics for each prediction
class EvalMetrics:
    def __init__(self, accuracy, precision, recall, runtime=0):
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.runtime = runtime
    
    # Output performance metrics in formatted table
    def __str__(self):
        timeOutput = f"{int(self.runtime)}ms" if (self.runtime != 0) else "-"
        tableFormat = "+--------------+-----------------+\n"
        tableFormat += f"| Accuracy     | {self.accuracy:^15.4f} |\n"
        tableFormat += f"| Precision    | {self.precision:^15.4f} |\n"
        tableFormat += f"| Recall       | {self.recall:^15.4f} |\n"
        tableFormat += f"| Runtime      | {timeOutput:^15s} |\n"
        tableFormat += "+--------------+-----------------+\n"
        return tableFormat

    # Calculate performance metrics for prediction
    @staticmethod
    def calculate(predMask, gtMask):
        # Convert masks to binary format (0 or 1)
        predMask = np.array(predMask, dtype=np.uint8)
        gtMask = np.array(gtMask, dtype=np.uint8)
    
        # Calculate True Positives (TP), False Positives (FP), False Negatives (FN)
        tp = np.count_nonzero(np.logical_and(predMask == 1, gtMask == 1))
        fp = np.count_nonzero(np.logical_and(predMask == 1, gtMask == 0))
        tn = np.count_nonzero(np.logical_and(predMask == 0, gtMask == 0))
        fn = np.count_nonzero(np.logical_and(predMask == 0, gtMask == 1))
        
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        evalResults = EvalMetrics(accuracy, precision, recall)
        
        # EvalMetrics.visualize(predMask, gtMask)

        return evalResults

    # Calculates the average metrics given a list of EvalMetrics
    @staticmethod
    def aggregate(resultsList):
        if not resultsList:
            return "Empty results list...\n"
    
        totalAccuracy = 0.0
        totalPrecision = 0.0
        totalRecall = 0.0
        totalRuntime = 0
    
        numResults = len(resultsList)
    
        for result in resultsList:
            totalAccuracy += result.accuracy
            totalPrecision += result.precision
            totalRecall += result.recall
            totalRuntime += result.runtime
    
        avgMetrics = EvalMetrics(
            totalAccuracy / numResults,
            totalPrecision / numResults, 
            totalRecall / numResults,
            totalRuntime / numResults)
    
        return avgMetrics
    
    # Visualize prediction and ground truth masks
    @staticmethod
    def visualize(predMask, gtMask):
        pt.figure()
        pt.subplot(3, 2, 1)
        pt.imshow(predMask*255, cmap="gray")
        pt.title("Prediction")
        pt.subplot(3, 2, 2)
        pt.imshow(gtMask*255, cmap="gray")
        pt.title("Ground Truth")
        pt.subplot(3, 2, 3)
        pt.imshow(np.logical_and(predMask == 1, gtMask == 1)*255, cmap="gray")
        pt.title("True Positive")
        pt.subplot(3, 2, 4)
        pt.imshow(np.logical_and(predMask == 1, gtMask == 0)*255, cmap="gray")
        pt.title("False Positive")
        pt.subplot(3, 2, 5)
        pt.imshow(np.logical_and(predMask == 0, gtMask == 0)*255, cmap="gray")
        pt.title("True Negative")
        pt.subplot(3, 2, 6)
        pt.imshow(np.logical_and(predMask == 0, gtMask == 1)*255, cmap="gray")
        pt.title("False Negative")
        pt.waitforbuttonpress(0) 
        pt.close()

# -------------------------------------------------------------

