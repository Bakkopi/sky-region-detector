import os
import random
import cv2
import shutil

# -------- Utility functions for loading images --------

# Return a list of random images from a directory
def loadRandomImages(directoryPath, numImagesToLoad, seed=333):
    print(os.path.basename(os.path.normpath(directoryPath)))
    imagePaths = os.listdir(directoryPath)
    random.seed(seed)
    random.shuffle(imagePaths)

    selectedImagePaths = imagePaths[:numImagesToLoad]

    images = []
    num = 0
    for imagePath in selectedImagePaths:
        fullImagePath = os.path.join(directoryPath, imagePath)
        image = cv2.imread(fullImagePath)
        if image is not None:
            images.append(image)
            num += 1
            print(f"Loaded {num} images...")
    print()
    return images

def loadAllImages(directoryPath):
    images = []

    # Get a list of subdirectories within the given directory
    subdirectories = [subdir for subdir in os.listdir(directoryPath) if os.path.isdir(os.path.join(directoryPath, subdir))]
    
    for subdir in subdirectories:
        subdirPath = os.path.join(directoryPath, subdir)
    
        # List all image files in the subdirectory
        imageFiles = [file for file in os.listdir(subdirPath) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
        # Read and append images to the main list
        for imageFile in imageFiles:
            imagePath = os.path.join(subdirPath, imageFile)
            image = cv2.imread(imagePath)
            if image is not None:
                images.append((image, imageFile))
    
    return images

def loadAndSave(directoryPath, numImagesToLoad, outputDirectory, seed=333):
    imagePaths = os.listdir(directoryPath)
    random.seed(seed)
    random.shuffle(imagePaths)

    selectedImagePaths = imagePaths[:numImagesToLoad]

    images = []
    num = 0
    for imagePath in selectedImagePaths:
        fullImagePath = os.path.join(directoryPath, imagePath)
        image = cv2.imread(fullImagePath)
        if image is not None:
            images.append(image)
            num += 1
            print(f"Copied {num} images...")

            # Save the loaded image to the output directory with the same name
            outputFilePath = os.path.join(outputDirectory, os.path.basename(imagePath))
            cv2.imwrite(outputFilePath, image)

    return images

def copyImagesToNewDir(sourceDir, destinationDir, numImagesToCopy):
    # Get a list of subdirectories within the source directory
    subdirectories = [subdir for subdir in os.listdir(sourceDir) if os.path.isdir(os.path.join(sourceDir, subdir))]

    # Create the destination directory if it doesn't exist
    os.makedirs(destinationDir, exist_ok=True)

    # Loop through each subdirectory and copy random images
    for subdir in subdirectories:
        sourceSubdir = os.path.join(sourceDir, subdir)
        destinationSubdir = os.path.join(destinationDir, subdir)
        os.makedirs(destinationSubdir, exist_ok=True)

        # List all images in the subdirectory
        imageFiles = [file for file in os.listdir(sourceSubdir) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]

        # Select 25 random images (if available)
        numImages = min(len(imageFiles), numImagesToCopy)
        random.seed(234)
        randomImages = random.sample(imageFiles, numImages)

        # Copy the random images to the destination subdirectory
        for imageFile in randomImages:
            sourceImagePath = os.path.join(sourceSubdir, imageFile)
            destinationImagePath = os.path.join(destinationSubdir, imageFile)
            shutil.copy(sourceImagePath, destinationImagePath)
            


