import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras import backend as K
from skimage.transform import resize
from keras.models import load_model

def visualize_activation_maps(model, input_image, layer_name):
    # Creating a sub-model that outputs the activation maps of the specified layer
    activation_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    
    # Getting the activation maps for the input image
    activations = activation_model.predict(input_image)
    
    # Plotting the activation maps
    fig, axes = plt.subplots(nrows=4, ncols=8, figsize=(10, 10))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(activations[0, :, :, i], cmap='jet')
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def procesImage(img):
    img = tf.image.resize(img, (200, 200))
    img = np.expand_dims(img/255., axis=0)
    return img

model = load_model("Cypuka3.h5")
img = procesImage(cv2.imread('newData//realism3.jpg'))
prediction = model.predict(img)[0]
prob = [round(x, 3) for x in prediction]
print(prob)

layer_name = 'conv2d'
visualize_activation_maps(model, img, layer_name)
layer_name = 'conv2d_1'
visualize_activation_maps(model, img, layer_name)
layer_name = 'conv2d_2'
visualize_activation_maps(model, img, layer_name)

gb_model = Model(
    inputs = [model.inputs],    
    outputs = [model.get_layer('conv2d_2').output]
)
layer_dict = [layer for layer in gb_model.layers[1:] if hasattr(layer,'activation')]

@tf.custom_gradient
def guidedRelu(x):
  def grad(dy):
    return tf.cast(dy>0,"float32") * tf.cast(x>0, "float32") * dy
  return tf.nn.relu(x), grad

for layer in layer_dict:
  if layer.activation == tf.keras.activations.relu:
    layer.activation = guidedRelu

with tf.GradientTape() as tape:
    inputs = tf.cast(img, tf.float32)
    tape.watch(inputs)
    outputs = gb_model(inputs)[0]
grads = tape.gradient(outputs,inputs)[0]

weights = tf.reduce_mean(grads, axis=(0, 1))
grad_cam = np.ones(outputs.shape[0: 2], dtype = np.float32)
for i, w in enumerate(weights):
    grad_cam += w * outputs[:, :, i]

imgg = cv2.imread('newData//realism3.jpg')
imgg = tf.image.resize(imgg, (200, 200))

grad_cam_img = cv2.resize(grad_cam.numpy(), (200, 200))
grad_cam_img = np.maximum(grad_cam_img, 0)
heatmap = (grad_cam_img - grad_cam_img.min()) / (grad_cam_img.max() - grad_cam_img.min())
grad_cam_img = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
output_image = cv2.addWeighted(imgg.numpy().astype(np.uint8), 0.5, grad_cam_img, 1, 0)

plt.imshow(output_image)
plt.axis("off")
plt.show()

guided_back_prop =grads
gb_viz = np.dstack((
            guided_back_prop[:, :, 0],
            guided_back_prop[:, :, 1],
            guided_back_prop[:, :, 2],
        ))       
gb_viz -= np.min(gb_viz)
gb_viz /= gb_viz.max()
    
imgplot = plt.imshow(gb_viz)
plt.axis("off")
plt.show()

guided_cam = np.maximum(grad_cam, 0)
guided_cam = guided_cam / np.max(guided_cam) # scale 0 to 1.0
guided_cam = resize(guided_cam, (200,200), preserve_range=True)
#pointwise multiplcation of guided backprop and grad CAM 
gd_gb = np.dstack((
        guided_back_prop[:, :, 0] * guided_cam,
        guided_back_prop[:, :, 1] * guided_cam,
        guided_back_prop[:, :, 2] * guided_cam,
    ))
imgplot = plt.imshow(gd_gb)
plt.axis("off")
plt.show()
