import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import numpy as np

# Load content and style images (see example in the attached colab).
content_name = 'japan2'
style_name = 'rococo'
content_image = plt.imread(f'newData\\{content_name}.jpg')
style_image = plt.imread(f'newData\\{style_name}.jpg')
# Convert to float32 numpy array, add batch dimension, and normalize to range [0, 1]. Example using numpy:
content_image = content_image.astype(np.float32)[np.newaxis, ...] / 255.
style_image = style_image.astype(np.float32)[np.newaxis, ...] / 255.
# Optionally resize the images. It is recommended that the style image is about
# 256 pixels (this size was used when training the style transfer network).
# The content image can be any size.
style_image = tf.image.resize(style_image, (256, 256))

# Load image stylization module.
hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

# Stylize image.
outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
stylized_image = outputs[0]

plt.imshow(stylized_image[0])
plt.show()