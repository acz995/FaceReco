import tensorflow as tf
import cv2
import json
import numpy as np
from matplotlib import pyplot as plt


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

print(tf.config.list_physical_devices('GPU'))
images = tf.data.Dataset.list_files('data\\images\\*.jpg', shuffle=False)
print(images.as_numpy_iterator().next())