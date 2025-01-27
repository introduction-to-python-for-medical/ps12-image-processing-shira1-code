from PIL import Image
import numpy as np
from scipy.signal import convolve2d

def load_image(file_path):
    image = Image.open(file_path)
    image_array = np.array(image)
    return image_array
   
def edge_detection(image_array):
    grayscale_image = np.mean(image_array, axis=2)
    kernelY = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    kernelx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    edged_image_x = convolve2d(grayscale_image, kernelx, mode='same', boundary='fill', fillvalue=0)
    edged_image_y = convolve2d(grayscale_image, kernelY, mode='same', boundary='fill', fillvalue=0)
    edgeMAG = np.sqrt(edged_image_x**2 + edged_image_y**2)
    return edgeMAG

    edge_detection(load_image('mypic.jpeg'))


