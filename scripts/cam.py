import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import tensorflow as tf
from scripts.utils import preprocess_image, get_class_activation_map

# Load model
model = load_model('models/cifar10_model.h5')

# Load and preprocess an example image
img_path = 'airplane.jpg'
img = preprocess_image(img_path)

# Generate CAM
class_idx = np.argmax(model.predict(img))
cam = get_class_activation_map(model, img, class_idx)

# Display CAM
plt.imshow(cam, cmap='jet', alpha=0.5)
plt.show()
