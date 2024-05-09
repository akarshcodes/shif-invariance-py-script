import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models

def build_optical_model(input_shape):
    inputs = layers.Input(shape=input_shape)
    distorted = layers.Lambda(lambda x: distort_images(x))(inputs)
    blurred = layers.GaussianNoise(0.1)(distorted)
    noise = layers.GaussianNoise(0.1)(inputs)
    concatenated = layers.Concatenate()([inputs, blurred, noise])
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(concatenated)
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv4)
    up1 = layers.UpSampling2D(size=(2, 2))(conv4)
    conv5 = layers.Conv2D(256, 3, activation='relu', padding='same')(up1)
    conv5 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv5)
    up2 = layers.UpSampling2D(size=(2, 2))(conv5)
    conv6 = layers.Conv2D(128, 3, activation='relu', padding='same')(up2)
    conv6 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv6)
    up3 = layers.UpSampling2D(size=(2, 2))(conv6)
    conv7 = layers.Conv2D(64, 3, activation='relu', padding='same')(up3)
    conv7 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv7)
    outputs = layers.Conv2D(3, 3, activation='sigmoid', padding='same')(conv7)
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

def distort_images(images):
    distorted_images = np.zeros_like(images)
    for i in range(images.shape[0]):
        distorted_images[i] = barrel_distortion(images[i])
    return distorted_images

def barrel_distortion(image):
    return np.roll(image, 10, axis=1)

def load_images_from_directory(directory):
    image_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.jpg')]
    images = [np.array(Image.open(f)) for f in image_files]
    return np.array(images)

def optical_image_enhancement(imperfect_images, perfect_images):
    imperfect_images = imperfect_images / 255.0
    perfect_images = perfect_images / 255.0
    model = build_optical_model(imperfect_images[0].shape)
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(imperfect_images, perfect_images, epochs=10, batch_size=32, validation_split=0.2)
    return model

imperfect_images = load_images_from_directory('imperfect jpg path')
perfect_images = load_images_from_directory('perfect_images render')
model = optical_image_enhancement(imperfect_images, perfect_images)
new_imperfect_image = np.array(Image.open('input img'))
enhanced_image = model.predict(np.expand_dims(new_imperfect_image, axis=0))