import numpy as np
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
from PIL import Image
from skimage.filters import median
from skimage.morphology import ball

def load_image(path):
    """Loads an image from the given path and returns it as a numpy array."""
    loaded_image = Image.open(path)
    return np.array(loaded_image)

# Load the image
image_path = 'Thailand.jpg'
image = load_image(image_path)
plt.imshow(image)
plt.axis("off")
plt.show()

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
edges = edge_detection(image)

# Apply a median filter to the original image
clean_image = median(image, ball(3))

# Save the edge-detected image
edge_image = Image.fromarray((edges * 255 / np.max(edges)).astype(np.uint8))
edge_image.save('my_edges.png')

# Plot the original, cleaned, and edge detection result
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(image)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Cleaned Image")
plt.imshow(clean_image)
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Edge Detection")
plt.imshow(edges, cmap="gray")
plt.axis("off")

plt.show()
