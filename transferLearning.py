import numpy as np 
import cv2
import PIL.Image as Image

import os 
import matplotlib.pylab as plt 


import tensorflow as tf 
import tensorflow_hub as hub
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


IMAGE_SHAPE = (224, 224)
classfier = tf.keras.Sequential(
    [
        hub.KerasLayer("https://www.kaggle.com/models/google/mobilenet-v2/frameworks/TensorFlow2/variations/tf2-preview-classification/versions/4", input_shape=IMAGE_SHAPE+(3,))

    ]
)

daisy = cv2.imread("D:\\pydata\\datasets\\flower_photos\\daisy\\5547758_eea9edfd54_n.jpg")
daisy= cv2.resize(daisy, (224, 224), interpolation = cv2.INTER_LINEAR)/255.0
daisy = np.reshape(daisy, (-1, 224, 224, 3))


# result = classfier.predict(dog)


# predict_label_index = np.argmax(result)

# print(predict_label_index)

# dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
# data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url,  cache_dir='.', untar=True)
# cache_dir indicates where to download data. I specified . which means current directory
# untar true will unzip it
import pathlib
data_dir = pathlib.Path("D:\\pydata\\datasets\\flower_photos\\")


image_count = len(list(data_dir.glob('**/*.jpg')))
print(image_count)
flowers_images_dict = {
    'roses': list(data_dir.glob('roses/*')),
    'daisy': list(data_dir.glob('daisy/*')),
    'dandelion': list(data_dir.glob('dandelion/*')),
    'sunflowers': list(data_dir.glob('sunflowers/*')),
    'tulips': list(data_dir.glob('tulips/*')),
}
flowers_labels_dict = {
    'roses': 0,
    'daisy': 1,
    'dandelion': 2,
    'sunflowers': 3,
    'tulips': 4,
}
X, y = [], []

for flower_name, images in flowers_images_dict.items():
    for image in images:
        img = cv2.imread(str(image))
        resized_img = cv2.resize(img,(224,224))
        X.append(resized_img)
        y.append(flowers_labels_dict[flower_name])
from sklearn.model_selection import train_test_split
X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


X_train_scaled = X_train / 255
X_test_scaled = X_test / 255

feature_extractor_model = "https://www.kaggle.com/models/google/mobilenet-v2/frameworks/TensorFlow2/variations/tf2-preview-feature-vector/versions/4"

pretrained_model_without_top_layer = hub.KerasLayer(
    feature_extractor_model, input_shape=(224, 224, 3), trainable=False)

num_of_flowers = 5

model = tf.keras.Sequential([
  pretrained_model_without_top_layer,
  tf.keras.layers.Dense(num_of_flowers)
])
model.compile(
  optimizer="adam",
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

history = model.fit(X_train_scaled, y_train, epochs=3)

plt.plot(history.history['accuracy'], label='accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

model.evaluate(X_test_scaled,y_test)
plt.show()

result = model.predict(daisy)


predict_label_index = np.argmax(result)

print(predict_label_index)


# Save the entire model to a HDF5 file.
# The '.h5' extension indicates that the model should be saved to HDF5.
model.save('my_model.h5')

# print(model.predict())
# flowers_labels_dict = {
#     'roses': 0,
#     'daisy': 1,
#     'dandelion': 2,
#     'sunflowers': 3,
#     'tulips': 4,
# }
flower_names  = ["roses", "daisy", "dandelion", "sunerflowers", "tulips"]

print(flower_names[result])