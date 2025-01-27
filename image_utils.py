import numpy as np
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from PIL import Image

def load_image(path):
    """Loads an image from the specified path.

    Args:
        path (str): The path to the image file.

    Returns:
        numpy.ndarray: The image as a NumPy array.
    """
    load_image = Image.open(path)  # Indented to be part of the function
    image = np.array(load_image)
    #plt.imshow(load_image); #This line is unnecessary within the function and may be causing issues remove it.
    return image #return the image array

#Now you can load your image like this:
#image = load_image('Thailand.jpg')
#plt.imshow(image); # Now display the image after loading



def edge_detection(image_array):
    """
    Perform edge detection on an image array.

    Args:
        image_array (numpy.ndarray): 3-channel color image array.

    Returns:
        numpy.ndarray: Edge magnitude array after detecting edges.
    """
    # Convert the 3-channel image to grayscale
    grayscale = np.mean(image_array, axis=2)

    # Define the vertical and horizontal filters
    kernelY = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]
    ])
    kernelX = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ])

    # Apply the filters to detect edges
    edgeY = convolve(grayscale, kernelY, mode='constant', cval=0.0)
    edgeX = convolve(grayscale, kernelX, mode='constant', cval=0.0)

    # Combine edges in both directions
    edgeMAG = np.sqrt(edgeX**2 + edgeY**2)

    return edgeMAG

# Perform edge detection
# Assuming 'image' is loaded using the load_image function
# image = load_image('Thailand.jpg')  # Load the image using your function
edges = edge_detection(image)

# Plot the original image and the edge detection result
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Edge Detection")
plt.imshow(edges, cmap="gray")
plt.axis("off")

plt.show()
