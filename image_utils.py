from PIL import Image
import numpy as np
from scipy.signal import convolve2d

def load_image(path):
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

load_image = Image.open('Thailand.jpg')
image = np.array(load_image)
plt.imshow(load_image);

def edge_detection(image):
    pass # Replace the `pass` with your code
