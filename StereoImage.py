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

    #Validating that batch inputs are odd and above 5
    if block_size % 2 == 0:
        block_size += 1

    if block_size < 5:
        block_size = 5

    # Calculate the disparity map with the updated parameters
    #disparity_map = getDisparityMap(imgL, imgR, num_disparities, block_size)
    disparity_map = getDisparityMap(edgesL, edgesR, num_disparities, block_size)

    #Normalise the disparity map
    disparityImg = np.interp(disparity_map, (disparity.min(), disparity.max()), (0.0, 1.0))
    # Display the resulting disparity map
    cv2.imshow('Disparity', disparityImg)
    #cv2.imshow('Disparity', disparity_map)

# ================================================
#
def plot(disparity):
    shape = disparity.shape
    f = 5806.559 #pixels
    doffs = 114.291 #pixels
    baseline = 174.019 #mm
    fmil = 41.81 #mm calculated earlier
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
                #x/X = f/Z, X = x * Z/f

                X = Z * (r) / fmil
                Y = Z * (c) / fmil

                #X is calculated in pixels so I use the conversion rate of 0.0072 that we found earlier to convert to mm.
                x += [X*0.0072]
                y += [Y*0.0072]
                z += [Z]

    # Plt depths
    ax = plt.axes(projection ='3d')
    ax.scatter(x, y, z, 'green')

    #Used to display certain axis
    #front on view
    #ax.view_init(elev=90, azim=0)
    #top down view
    #ax.view_init(elev=0, azim=0)
    #side view
    #ax.view_init(elev=0, azim=270)

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
    filename = 'UmL.png'
    imgL = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    #
    if imgL is None:
        print('\nError: failed to open {}.\n'.format(filename))
        sys.exit()

    #Resize image as suggested
    imgL = cv2.resize(imgL, (740, 505))

    #Canny edge detection code
    # Apply Gaussian blur
    #blurL = cv2.GaussianBlur(imgL, (3, 3), 0)

    # Apply Canny edge detection
    edgesL = cv2.Canny(imgL, 50, 100)

    # Load right image
    filename = 'UmR.png'
    imgR = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    if imgR is None:
        print('\nError: failed to open {}.\n'.format(filename))
        sys.exit()

    #Reszing images as suggested
    imgR= cv2.resize(imgR, (740, 505))

    #blurR = cv2.GaussianBlur(imgR, (5, 5), 0)
    # Apply Canny edge detection
    edgesR = cv2.Canny(imgR, 50, 100)

    # Display the resulting image
    cv2.imshow('Edges', edgesR)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Create a window to display the image in
    cv2.namedWindow('Disparity', cv2.WINDOW_NORMAL)

    #Creating trackbars for our parameters
    cv2.createTrackbar('Num Disparities', 'Disparity', 1, 16, update_disparity_map)
    cv2.createTrackbar('Block Size', 'Disparity', 5, 255, update_disparity_map)

    # Get disparity map
    #disparity = getDisparityMap(imgL, imgR, 64, 5)
    disparity = getDisparityMap(edgesL, edgesR, 80, 9)

    # Normalise for display
    disparityImg = np.interp(disparity, (disparity.min(), disparity.max()), (0.0, 1.0))

    # Show result
    cv2.imshow('Disparity', disparityImg)

    # Show 3D plot of the scene
    plot(disparity)

    # Wait for spacebar press or escape before closing,
    # otherwise window will close without you seeing it
    while True:
        key = cv2.waitKey(1)
        if key == ord(' ') or key == 27:
            break

    cv2.destroyAllWindows()
