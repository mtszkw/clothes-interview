import argparse
import requests

import numpy as np
from skimage.io import imread
import tensorflow as tf

from preprocessing import preprocess_func
from inference_utils import extract_model, prediction_vector_to_label


def load_image_from_url(url):
    input_image = imread(url)
    input_image = preprocess_func(input_image)
    input_image = np.expand_dims(input_image, axis=0)
    print(input_image.shape)
    return input_image


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Clothes Classification')
    parser.add_argument('-u', '--url', action='store', dest='url', default=None, help='<Required> URL link', required=True)

    inference_image_url = parser.parse_args().url
    image = load_image_from_url(inference_image_url)

    extract_model('./bin/mobilenet_fashion_clothes_clf.tar.gz')
    inference_model = tf.keras.models.load_model('./content/mobilenet_fashion_clothes_clf/')

    y_pred = inference_model.predict(image, batch_size=1)
    label, score = prediction_vector_to_label(y_pred)
    print(f'Predicted class: {label} (confidence score: {score:.5f})')
