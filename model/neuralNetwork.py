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
    result = tf.matmul(input_tensor, input_tensor, transpose_b=True)
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    num = input_tensor.shape[1] * input_tensor.shape[2]
    result = result/num
    return result


class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = vggLayers(style_layers + content_layers)
        self.vgg.trainable = False
        self.style_layers = style_layers
        self.content_layers = content_layers

    def call(self, inputs):
        inputs = inputs * 255
        inputs = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(inputs)
        style_outputs, content_outputs = (
            outputs[:len(self.style_layers)], outputs[len(self.style_layers):])
        style_outputs = [gram_matrix(style_output)
                         for style_output in style_outputs]
        content_dict = {layer_name: value for layer_name,
                        value in zip(self.content_layers, content_outputs)}
        style_dict = {layer_name: value for layer_name,
                      value in zip(self.style_layers, style_outputs)}
        return {"content": content_dict, "style": style_dict}


extractor = StyleContentModel(style_layers, content_layers)
content_values = extractor(contentImage)['content']
style_values = extractor(styleImage)['style']

for name, output in style_values.items():
    print(name)
    print(output.shape)

for name, output in content_values.items():
    print(name)
    print(output.shape)
