import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

style_path = tf.keras.utils.get_file(
    "styleImage1.jpg", "https://wallpaperaccess.com/full/320455.jpg")
content_path = tf.keras.utils.get_file(
    "contentImage1.jpg", "https://fullhdwall.com/wp-content/uploads/2016/07/Free-Mountains-Landscape.jpg")


def readImage(image_path):
    maxSize = 480
    img = tf.io.read_file(style_path)
    img = tf.io.decode_image(img)
    img = tf.image.convert_image_dtype(img, tf.float32)
    ratio = maxSize/img.shape[0]
    img = tf.image.resize(
        img, [int(img.shape[0]*ratio), int(img.shape[1]*ratio)])
    img = img[tf.newaxis, :]
    return img


styleImage = readImage(style_path)
contentImage = readImage(content_path)

inputData = tf.keras.applications.vgg19.preprocess_input(contentImage)
inputData = tf.image.resize(inputData, [224, 224])
vggModel = tf.keras.applications.VGG19(include_top=False)
for layer in vggModel.layers:
    print(layer.name)

style_layers = ['block1_conv1', 'block2_conv1',
                'block3_conv1', 'block4_conv1', 'block5_conv1']
content_layers = ['block4_conv2']


def vggLayers(layer_names):
    vggModel = tf.keras.applications.VGG19(include_top=False)
    vggModel.trainable = False
    outputs = [vggModel.get_layer(name).output for name in layer_names]
    model = tf.keras.Model([vggModel.input], outputs)
    return model


def gram_matrix(input_tensor):
    result = tf.matmul(input_tensor, tf.transpose(input_tensor))
    num = input_tensor.shape[1] * input_tensor.shape[2]
    result = result/num
    return result
