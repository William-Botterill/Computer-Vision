import cv2
import numpy as np
import copy
import sys
from matplotlib import pyplot as plt

# Open JPEG image
img = cv2.imread('kitty.bmp')
cv2.imshow('Original Image', img)
dimensions = img.shape;

img_bordered = cv2.copyMakeBorder(src=img, top=1, bottom=1, left=1, right=1, borderType=cv2.BORDER_CONSTANT)
cv2.imshow('Bordered Image', img_bordered)
img = img_bordered.astype(int)

cv2.imwrite('BorderImage.jpg', img)

#For each output image I pre-assign memory as a numpy array of zeros
#This array is set to be of type int to ensure values don't overflow and cause error

output = np.zeros(dimensions, dtype = int)
output = output.astype(np.int)

weighted_output = np.zeros(dimensions, dtype = int)
weighted_output = weighted_output.astype(np.int)

Sobel_yOutput = np.zeros(dimensions, dtype = int)
Sobel_yOutput = Sobel_yOutput.astype(np.int)

Sobel_xOutput = np.zeros(dimensions, dtype = int)
Sobel_xOutput = Sobel_xOutput.astype(np.int)

Sobel_Output = np.zeros(dimensions, dtype = int)
Sobel_Output = Sobel_Output.astype(np.int)

SobelWeighted_yOutput = np.zeros(dimensions, dtype = int)
SobelWeighted_yOutput = SobelWeighted_yOutput.astype(np.int)

SobelWeighted_xOutput = np.zeros(dimensions, dtype = int)
SobelWeighted_xOutput = SobelWeighted_xOutput.astype(np.int)

SobelWeighted_Output = np.zeros(dimensions, dtype = int)
SobelWeighted_Output = SobelWeighted_Output.astype(np.int)

for i in range(1, dimensions[0]-1):
    for j in range(1, dimensions[1]-1):
        #Calculating the sum of the pixels in the 3x3 grid surrounding pixel [i,j]
        total = img[i-1,j-1] + img[i-1,j] + img[i-1,j+1] + img[i,j-1] + img[i,j] + img[i,j+1] + img[i+1,j-1] + img[i+1,j] + img[i+1,j+1]
        #Normalising the result
        output[i,j] = total/9

        #Calculating the weighted sum of the pixels in the 3x3 grid surrounding pixel [i,j]
        weighted_total = 0.5*img[i-1,j-1] + img[i-1,j] + 0.5*img[i-1,j+1] + img[i,j-1] + 2*img[i,j] + img[i,j+1] + 0.5*img[i+1,j-1] + img[i+1,j] + 0.5*img[i+1,j+1]
        #Normalising the result
        weighted_output[i,j] = weighted_total/8

        #Sobel Kernel for y-direction edge detection (derivative in the x direction)
        Sobel_yTotal = -1*img[i-1,j-1] + img[i-1,j+1] + -2*img[i,j-1] + 2*img[i,j+1] + -1*img[i+1,j-1] + img[i+1,j+1]
        #Normalising the result
        Sobel_yOutput[i,j] = Sobel_yTotal

        #Sobel Kernel for x-direction edge detection (derivative in the y direction)
        Sobel_xTotal = -1*img[i-1,j-1] + -1*img[i-1,j+1] + -2*img[i-1,j] + 2*img[i+1,j] + img[i+1,j-1] + img[i+1,j+1]
        #Normalising the result
        Sobel_xOutput[i,j] = Sobel_xTotal

        #For the edge strength Image
        Sobel_total = ((Sobel_xTotal)**2 + (Sobel_yTotal)**2)**0.5
        Sobel_Output[i,j] = Sobel_total

for i in range(1, dimensions[0]-1):
    for j in range(1, dimensions[1]-1):

        #Sobel Kernel for y-direction edge detection (derivative in the x direction)
        SobelWeighted_yTotal = -1*(weighted_output[i-1,j-1]) + (weighted_output[i-1,j+1]) + -2*(weighted_output[i,j-1]) + 2*(weighted_output[i,j+1]) + -1*(weighted_output[i+1,j-1]) + (weighted_output[i+1,j+1])
        #Normalising the result
        SobelWeighted_yOutput[i,j] = SobelWeighted_yTotal

        #Sobel Kernel for x-direction edge detection (derivative in the y direction)
        SobelWeighted_xTotal = -1*(weighted_output[i-1,j-1]) + -1*(weighted_output[i-1,j+1]) + -2*(weighted_output[i-1,j]) + 2*(weighted_output[i+1,j]) + (weighted_output[i+1,j-1]) + (weighted_output[i+1,j+1])
        #Normalising the result
        SobelWeighted_xOutput[i,j] = SobelWeighted_xTotal

        #For the edge strength Image
        SobelWeighted_total = ((SobelWeighted_xTotal)**2 + (SobelWeighted_yTotal)**2)**0.5
        SobelWeighted_Output[i,j] = SobelWeighted_total


#For each created image the output is clipped (to ensure values don't exceed 255)
#Images are then written to memory as a jpg

output = np.clip(output, 0, 255)
cv2.imwrite('kittyAverage.jpg', output)

weighted_output = np.clip(weighted_output, 0, 255)
cv2.imwrite('kittyWeightAverage.jpg', weighted_output)

Sobel_yOutput = np.clip(Sobel_yOutput, 0, 255)
cv2.imwrite('SobelYOutput.jpg', Sobel_yOutput)

Sobel_xOutput = np.clip(Sobel_xOutput, 0, 255)
cv2.imwrite('SobelXOutput.jpg', Sobel_xOutput)

Sobel_Output = np.clip(Sobel_Output, 0, 255)
cv2.imwrite('SobelOutput.jpg', Sobel_Output)

SobelWeighted_yOutput = np.clip(SobelWeighted_yOutput, 0, 255)
cv2.imwrite('WeightedSobelYOutput.jpg', SobelWeighted_yOutput)

SobelWeighted_xOutput = np.clip(SobelWeighted_xOutput, 0, 255)
cv2.imwrite('WeightedSobelXOutput.jpg', SobelWeighted_xOutput)

SobelWeighted_Output = np.clip(SobelWeighted_Output, 0, 255)
cv2.imwrite('WeightedSobelOutput.jpg', SobelWeighted_Output)

cat_SobelOutput = cv2.imread('SobelOutput.jpg')
weighted_SobelOutput = cv2.imread('WeightedSobelOutput.jpg')

#Code used to visualise the histograms for each output image, used to help determine threshold values

# Calculate the histogram
hist = cv2.calcHist([cat_SobelOutput], [0], None, [256], [0, 256])
#hist = cv2.calcHist([weighted_SobelOutput], [0], None, [256], [0, 256])
hist = hist.reshape(256)

# Plot histogram
plt.bar(np.linspace(0,255,256), hist)
plt.title('Histogram')
plt.ylabel('Frequency')
plt.xlabel('Grey Level')
plt.show()



# Threshold manually at intensity level 180 according to histogram results
_, output = cv2.threshold(cat_SobelOutput, 180, 255, cv2.THRESH_BINARY)
cv2.imwrite('SobelThres.jpg', output)

_, output2 = cv2.threshold(weighted_SobelOutput, 165, 255, cv2.THRESH_BINARY)
cv2.imwrite('WeightedSobelThres.jpg', output2)

difference = np.zeros(dimensions, dtype = int)
difference = difference.astype(np.int)

for i in range(1, dimensions[0]-1):
    for j in range(1, dimensions[1]-1):
        diff = output[i,j] - output2[i,j]
        if diff.any() < 0:
            diff = - diff
        difference[i,j] = diff

difference = np.clip(difference, 0, 255)
cv2.imwrite('difference.jpg', difference)

cat_edgeThres = cv2.imread('SobelThres.jpg')
cv2.imshow('Gradient Thresholded Image', cat_edgeThres)

#Below code used to display images on the screen for testing purposes

weighted_edgeThres = cv2.imread('WeightedSobelThres.jpg')
cv2.imshow('Weighted Gradient Thresholded Image', weighted_edgeThres)

cat_blur = cv2.imread('kittyAverage.jpg')
cv2.imshow('Average Kernal Image', cat_blur)

imgWeighted = cv2.imread('kittyWeightAverage.jpg')
cv2.imshow('Weighted Average Kernal Image', imgWeighted)

cat_SobelYOutput = cv2.imread('SobelYOutput.jpg')
cv2.imshow('Sobel Y Output on Original', cat_SobelYOutput)

cat_SobelXOutput = cv2.imread('SobelXOutput.jpg')
cv2.imshow('Sobel X Output on Original', cat_SobelXOutput)

cat_SobelOutput = cv2.imread('SobelOutput.jpg')
cv2.imshow('Image Gradient Output on Original', cat_SobelOutput)

weighted_SobelYOutput = cv2.imread('WeightedSobelYOutput.jpg')
cv2.imshow('Weighted Sobel Y Output on weighted average image', weighted_SobelYOutput)

weighted_SobelXOutput = cv2.imread('WeightedSobelXOutput.jpg')
cv2.imshow('Weighted Sobel X Output on weighted average image', weighted_SobelXOutput)

weighted_SobelOutput = cv2.imread('WeightedSobelOutput.jpg')
cv2.imshow('Image Gradient Output on Weighted Average Image', weighted_SobelOutput)

# Wait for spacebar press before closing,
# otherwise window will close without you seeing it
while True:
    if cv2.waitKey(1) == ord(' '):
        break

cv2.destroyAllWindows()
