import numpy as np
import tensorflow as tf
from keras.preprocessing import image

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(32, 32))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0

def get_class_activation_map(model, img, class_idx):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer('conv2d_1').output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img)
        loss = predictions[:, class_idx]

    output = conv_outputs[0]
    grads = tape.gradient(loss, conv_outputs)[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = np.zeros(output.shape[:2], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    cam = tf.image.resize(cam, (32, 32)).numpy()
    return cam
