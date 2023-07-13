import numpy as np
import cv2
import sys
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt

# ================================================
#
def getDisparityMap(imL, imR, numDisparities, blockSize):
    stereo = cv2.StereoBM_create(numDisparities=numDisparities, blockSize=blockSize)

    disparity = stereo.compute(imL, imR)
    disparity = disparity - disparity.min() + 1 # Add 1 so we don't get a zero depth, later
    disparity = disparity.astype(np.float32) / 16.0 # Map is fixed point int with 4 fractional bits

    return disparity # floating point image
# ================================================

def update_disparity_map(*args):
    # Get the current trackbar values
    num_disparities = cv2.getTrackbarPos('Num Disparities', 'Disparity')
    #To ensure num_disparities is a multiple of 16 as required
    num_disparities = num_disparities * 16;
    block_size = cv2.getTrackbarPos('Block Size', 'Disparity')
    k_size = cv2.getTrackbarPos('k', 'Disparity')

    #Validating that batch inputs are odd and above 5
    if block_size % 2 == 0:
        block_size += 1

    if block_size < 5:
        block_size = 5

    # Calculate the disparity map with the updated parameters
    disparity_map = getDisparityMap(imgL, imgR, num_disparities, block_size)

    disparity_map = np.interp(disparity_map, (disparity_map.min(), disparity_map.max()), (0.0, 1.0))
    depthMap = disparity_map.copy()
    shape = disparity_map.shape

    girlOut = imgLC.copy()
    for r in range(shape[0]):
        for c in range(shape[1]):
            d = disparity_map[r][c] #pixels
            if d == 0:
                depth = 255
            else:
                depth = 1/(d + k_size/255)

            depthMap[r][c] = depth
            if depth > 50:
                pixel = imgLC[r][c]
                # Convert the pixel to grayscale
                gray_value = 0.299 * pixel[2] + 0.587 * pixel[1] + 0.114 * pixel[0]
                girlOut[r][c] = gray_value

    cv2.imwrite('output.png', girlOut)
    cv2.imwrite('depthmap.png', depthMap)

    output = cv2.imread('output.png')
    cv2.imshow('Output', output)
    depthOutput = cv2.imread('depthmap.png')
    cv2.imshow('Depth', depthOutput)


    cv2.normalize(disparity_map, disparity_map, 0, 255, cv2.NORM_MINMAX)
    # Display the resulting disparity map
    #cv2.imshow('Disparity', disparityImg)
    cv2.imshow('Disparity', disparity_map)


# ================================================
#
def plot(disparity):
    shape = disparity.shape
    f = 5806.559 #pixels
    doffs = 114.291 #pixels
    baseline = 174.019 #mm
    fmil = 41.81 #mm calculated earlier
    #print(shape[0])
    # This just plots some sample points.  Change this function to
    # plot the 3D reconstruction from the disparity map and other values
    x = []
    y = []
    z = []
    for r in range(shape[0]):
        for c in range(shape[1]):
            #print(disparity[r][c])
            d = disparity[r][c] #pixels
            if d >  1:

                Z = baseline * f/(d + doffs)

                X = Z * r / fmil
                Y = Z * c / fmil

                x += [X*0.0072]
                y += [Y*0.0072]
                z += [Z]

    # Plt depths
    ax = plt.axes(projection ='3d')
    ax.scatter(x, y, z, 'green')

    # Labels
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.savefig('myplot.pdf', bbox_inches='tight') # Can also specify an image, e.g. myplot.png
    plt.show()
# ================================================
#
if __name__ == '__main__':

    # Load left image
    filename = 'girlL.png'
    imgL = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite('greyGirlL.png',imgL)

    if imgL is None:
        print('\nError: failed to open {}.\n'.format(filename))
        sys.exit()

    filename = 'girlR.png'
    imgR = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite('greyGirlR.png',imgR)
    if imgR is None:
        print('\nError: failed to open {}.\n'.format(filename))
        sys.exit()

    # Create a window to display the image in
    cv2.namedWindow('Disparity', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Output', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Depth', cv2.WINDOW_NORMAL)

    imgLC = cv2.imread('girlL.png')
    #Creating trackbars for our parameters
    cv2.createTrackbar('Num Disparities', 'Disparity', 1, 16, update_disparity_map)
    cv2.createTrackbar('Block Size', 'Disparity', 5, 255, update_disparity_map)
    cv2.createTrackbar('k', 'Disparity', 0, 255, update_disparity_map)

    # Get disparity map
    disparity = getDisparityMap(imgL, imgR, 80, 33)
    cv2.imshow('Disparity', disparity)

    # Wait for spacebar press or escape before closing,
    # otherwise window will close without you seeing it
    while True:
        key = cv2.waitKey(1)
        if key == ord(' ') or key == 27:
            break

    cv2.destroyAllWindows()
