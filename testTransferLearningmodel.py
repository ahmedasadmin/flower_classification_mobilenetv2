import numpy as np 
import cv2
import matplotlib.pylab as plt 
import tensorflow as tf
import tensorflow_hub as hub


path = "my_model.h5"
"""
    i ve used stack overflow solution and 
    fortunately, it is work fine
"""
model= tf.keras.models.load_model(
       (path),
       custom_objects={
           'KerasLayer':hub.KerasLayer
        }
)


imagePath = "D:\\pydata\\datasets\\flower_photos\\daisy\\5547758_eea9edfd54_n.jpg"


daisy = cv2.imread(imagePath)


# normalize input image pixels values between [0, 1]
daisy= cv2.resize(daisy, (224, 224), interpolation = cv2.INTER_LINEAR)/255
daisy = np.reshape(daisy, (-1, 224, 224, 3))

result = model.predict(daisy)
predict_label_index = np.argmax(result)
print(predict_label_index)
"""
    create list of labels and print output of prediction 
"""
flower_names  = ["roses", "daisy", "dandelion", "sunerflowers", "tulips"]
print(flower_names[predict_label_index])
