import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

SIZE = 224


def resize_image(img):
    img = tf.cast(img, tf.float32)
    img = tf.image.resize(img, (SIZE, SIZE))
    img = img / 255.0
    return img


inputfile = input()

model = load_model('network.h5')
img = load_img(inputfile)
img_array = img_to_array(img)
img_resized = resize_image(img_array)
img_expended = np.expand_dims(img_resized, axis=0)
prediction = model.predict(img_expended)[0][0]
result = tf.sigmoid(prediction)
pred_label = 'КОТ' if result < 0.5 else 'СОБАКА'
plt.figure()
plt.imshow(img)
plt.title(f'{pred_label} {result}')
plt.show()
