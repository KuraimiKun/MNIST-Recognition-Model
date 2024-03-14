import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Check if GPU is available
print("GPU Available:", tf.config.list_physical_devices('GPU'))