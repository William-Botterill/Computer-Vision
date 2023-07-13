import cv2
import numpy as np
import copy
import sys
from matplotlib import pyplot as plt
#import skimage.exposure as exposure


img = cv2.imread('kitty.bmp',cv2.IMREAD_GRAYSCALE)
dst = cv2.copyMakeBorder(img, 3, 3, 3, 3, cv2.BORDER_REFLECT_101)
cv2.imshow('border', dst)

#cv2.imshow('Original Image', img)
dimensions = dst.shape;



def HarrisPointsDetector():
    sobelx = cv2.Sobel(src=dst, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3) # Sobel Edge Detection on the X axis
    sobely = cv2.Sobel(src=dst, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3) # Sobel Edge Detection on the Y axis
    #sobelxy = cv2.Sobel(src=dst, ddepth=cv2.CV_16S, dx=1, dy=1, ksize=3) # Combined X and Y Sobel Edge Detection

    # optionally normalize to range 0 to 255 for proper display
    sobelx_norm = np.clip(sobelx, 0, 255)
    sobely_norm = np.clip(sobely, 0, 255)
    #sobelx_norm= exposure.rescale_intensity(sobelx, in_range='image', out_range=(0,255)).clip(0,255).astype(np.uint8)

    #sobely_norm= exposure.rescale_intensity(sobelx, in_range='image', out_range=(0,255)).clip(0,255).astype(np.uint8)

    # square
    sobelx2 = cv2.multiply(sobelx,sobelx)
    sobely2 = cv2.multiply(sobely,sobely)

    # add together and take square root
    sobel_magnitude = cv2.sqrt(sobelx2 + sobely2)

    # normalize to range 0 to 255 and clip negatives
    sobel_magnitude = np.clip(sobel_magnitude, 0, 255)
    #sobel_magnitude = exposure.rescale_intensity(sobel_magnitude, in_range='image', out_range=(0,255)).clip(0,255).astype(np.uint8)

    output = np.zeros(dimensions, dtype = int)
    output = output.astype(np.int)
    # Display Sobel Edge Detection Images

    #matrix = [[sobelx[i][j], 2],[5, 6]]
    #print(sobelx[50][50])
    for i in range(3, dimensions[0]-3):
        for j in range(3, dimensions[1]-3):
            M = [[0,0],[0,0]]
            for d in range(-1, 2):
                for e in range(-1,2):
                    #print("a")
                    temp = [[(sobelx[i+d][j+e])**2, (sobely[i+d][j+e] * sobelx[i+d][j+e])],
                            [(sobelx[i+d][j+e] * sobely[i+d][j+e]), (sobely[i+d][j+e])**2]]
                    #print(sobelx[i][j])
                    M[0][0] += temp[0][0] / 9
                    M[0][1] += temp[0][1] / 9
                    M[1][0] += temp[1][0] / 9
                    M[1][1] += temp[1][1] / 9

            #Trace is the sum of diagonal elements
            #determinant is calculate ad - bc
            #R = c(M) = det(M) - 0.05(trace(M))^2
            #R = 100
            #R = M[0][0] * M[1][1] - M[0][1] * M[1][0] - 0.05
            #print(M)
            R = (M[0][0] * M[1][1]) - (M[0][1] * M[1][0]) - 0.05 * (M[0][0] + M[1][1])**2
            output[i,j] = R

    output2 = np.zeros(dimensions, dtype = int)
    output2 = output2.astype(np.int)
    max = 0
    #output = cv2.copyMakeBorder(output, 2, 2, 2, 2, cv2.BORDER_REFLECT_101)
    for i in range(3, dimensions[0]-3):
        for j in range(3, dimensions[1]-3):
            temp = []
            for d in range(-3, 4):
                for e in range(-3,4):
                    temp += [output[i+d][j+e]]
                    #temp += [5];
            flag = True
            for k in range(49):
                if temp[24] < temp[k]:
                    flag = False
            if flag:
                output2[i,j] = temp[24]
                if temp[24] > max:
                    max = temp[24]

    output3 = output2
    outter2 = np.clip(output2, 0, 255)
    cv2.imwrite('MaximalOuter.jpg', outter2)
    maxOuter = cv2.imread('MaximalOuter.jpg')
    cv2.imshow('MaxOUter', maxOuter)

    total = 0
    below = 0
    for i in range(0, dimensions[0]):
        for j in range(0, dimensions[1]):
            total = total + 1
            if(output3[i,j] < 15000):
                output3[i,j] = 0
                below = below + 1

    print("total: ")
    print(total)
    print("Killed: ")
    print(below)

    outter3 = np.clip(output3, 0, 255)
    cv2.imwrite('CulledOut.jpg', outter3)
    culled = cv2.imread('CulledOut.jpg')
    cv2.imshow('Culled', culled)

    outter = np.clip(output, 0, 255)
    cv2.imwrite('outIm.jpg', outter)
    outIm = cv2.imread('outIm.jpg')
    cv2.imshow('output', outIm)
    #cv2.imwrite('')



    cv2.imshow('Gray', img)
    cv2.waitKey(0)
    cv2.imshow('Sobel X', sobelx_norm)
    cv2.waitKey(0)
    cv2.imshow('Sobel Y', sobely_norm)
    cv2.waitKey(0)
    cv2.imshow('Sobel X Y using Sobel() function', sobel_magnitude)
    cv2.waitKey(0)
    return

HarrisPointsDetector()




# Wait for spacebar press before closing,
# otherwise window will close without you seeing it
while True:
    if cv2.waitKey(1) == ord(' '):
        break

cv2.destroyAllWindows()
