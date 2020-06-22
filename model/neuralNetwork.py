import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image
import time

style_path = tf.keras.utils.get_file(
    "styleImage1.jpg", "https://wallpaperaccess.com/full/320455.jpg")
content_path = tf.keras.utils.get_file(
    "contentImage1.jpg", "https://images.wallpaperscraft.com/image/mountains_fog_dawn_153238_1920x1200.jpg")


def loadImage(image_path):
    maxSize = 480
    img = tf.io.read_file(image_path)
    img = tf.io.decode_image(img)
    img = tf.image.convert_image_dtype(img, tf.float32)
    ratio = maxSize/img.shape[0]
    img = tf.image.resize(
        img, (int(img.shape[0]*ratio), int(img.shape[1]*ratio)))
    img = img[tf.newaxis, :]
    return img


styleImage = loadImage(style_path)
contentImage = loadImage(content_path)


style_layers = ['block1_conv1', 'block2_conv1',
                'block3_conv1', 'block4_conv1', 'block5_conv1']
content_layers = ['block5_conv2']
num_style_layers = len(style_layers)
num_content_layers = len(content_layers)


def tensor_to_image(image):
    image = image*255
    image = np.array(image, dtype=np.uint8)
    if np.ndim(image) > 3:
        assert image.shape[0] == 1
        image = image[0]
    return PIL.Image.fromarray(image)


def vggLayers(layer_names):
    vggModel = tf.keras.applications.VGG19(include_top=False)
    vggModel.trainable = False
    outputs = [vggModel.get_layer(name).output for name in layer_names]
    model = tf.keras.Model([vggModel.input], outputs)
    return model


def gram_matrix(input_tensor):
    result = tf.linalg.einsum("bijc, bijd -> bcd", input_tensor, input_tensor)
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
        inputs = inputs * 255.0
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
results = extractor(contentImage)
content_values = extractor(contentImage)['content']
style_values = extractor(styleImage)['style']

image = tf.Variable(contentImage)
opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)


def style_content_loss(outputs, style_weight=1e-2, content_weight=1e4):
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n([tf.reduce_mean(
        (style_outputs[name] - style_values[name])**2) for name in style_outputs.keys()])
    style_loss *= style_weight/num_style_layers
    content_loss = tf.add_n([tf.reduce_mean(
        (content_outputs[name] - content_values[name])**2) for name in content_outputs.keys()])
    content_loss *= style_weight/num_content_layers
    style_content_loss = style_loss + content_loss
    return style_content_loss


@ tf.function()
def train_step(image):
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = style_content_loss(outputs)

    grad = tape.gradient(loss, image)
    opt.apply_gradients([(grad, image)])
    image.assign(tf.clip_by_value(image, 0.0, 1.0))


start = time.time()

epochs = 10
steps_per_epoch = 100

step = 0
for n in range(epochs):
    for m in range(steps_per_epoch):
        step += 1
        train_step(image)
    print("Train step: {}".format(step))

end = time.time()
print("Total time: {:.1f}".format(end-start))

image = tensor_to_image(image)
image.save('result.jpg')
