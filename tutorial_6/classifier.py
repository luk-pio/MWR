# Using attached reference and library documentation implement a image
# classifier.
#
# It should be possible to run it in two modes:
#
# Training when a cifar10 set is used to train and save network
# Classification when network is loaded from file and then a given image is
# classified
# I would also like to remind you that input image must be rescaled to same
# size as those in cifar10 to
#
# work with same network.

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from keras.datasets import cifar10
from argparse import ArgumentParser
from keras.utils import to_categorical
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential, load_model

CLASS_LABELS = ['plane', 'car', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck']


def load_data():
    (x_tr, y_tr), (x_tst, y_tst) = cifar10.load_data()
    y_tr = to_categorical(y_tr, len(CLASS_LABELS))
    y_tst = to_categorical(y_tst, len(CLASS_LABELS))
    return (x_tr, y_tr), (x_tst, y_tst)


def define_network():
    # FEATURE EXTRACTION
    first_convolution = Conv2D(32, (3, 3), activation='relu', padding='same',
                               name='first_convolution',
                               input_shape=(32, 32, 3))
    second_convolution = Conv2D(32, (3, 3), activation='relu', padding='same',
                                name='second_convolution')
    scaling_down = MaxPooling2D(pool_size=(2, 2))

    # CLASSIFICATION
    classifier_input = Flatten(name='classifier_input')
    classifier_hidden = Dense(512, activation='relu', name='classifier_hidden')
    overfitting_countermeasure = Dropout(0.5,
                                         name='overfitting_countermeasure')
    classifier_output = Dense(len(CLASS_LABELS), activation='softmax',
                              name='classifier_output')

    network = Sequential([
            first_convolution,
            second_convolution,
            scaling_down,
            classifier_input,
            classifier_hidden,
            overfitting_countermeasure,
            classifier_output
    ])
    network.summary()
    return network


def train_network(x_train, y_train, x_test, y_test, network, epochs,
                  batch_size=32):
    print(x_train.shape)
    network.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
    network.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                validation_data=(x_test, y_test))

    return network


def save_network(network, path=".", name="network"):
    dirs = Path(path)
    if not dirs.is_dir():
        name = dirs.name
        dirs = dirs.parent
        if not dirs.is_dir():
            os.makedirs(dirs)

    file_count = 0
    while True:
        file_name = f'{name}_{file_count}.m5'
        target_path = dirs / file_name
        if target_path.is_file():
            file_count += 1
            continue
        network.save(target_path)
        print(f'Saved trained network at {target_path}')
        break


def load_file(load_func, path):
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"No file found at path {p}")
    return load_func(path)


def to_size(img):
    h, w = img.shape[:2]
    if h != w:
        raise ValueError("The image must have a 1:1 aspect ratio!")
    return cv.resize(img, (32, 32), interpolation=cv.INTER_AREA)


def classify(img_path, network_path):
    network = load_file(load_model, network_path)
    img = load_file(cv.imread, img_path)
    img = to_size(img)
    img = np.array([img])
    prediction = network.predict(img)
    print(prediction)
    show_prediction(prediction)


def show_prediction(prediction):
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(20, 10))
    indices = np.arange(len(CLASS_LABELS))
    margin = 0.05
    p = axes
    p.barh(indices + margin, prediction[0], 1)
    p.set_yticks(indices + margin)
    p.set_yticklabels(CLASS_LABELS[::-1])
    p.set_xticks([0, 0.5, 1.0])
    plt.show()


def train(epochs, network_path):
    (x_tr, y_tr), (x_tst, y_tst) = load_data()
    network = define_network()
    network = train_network(x_tr, y_tr, x_tst, y_tst, network, epochs)
    save_network(network, network_path)


def parse_arguments():
    desc = 'Image scaling software for MWR classes.'
    parser = ArgumentParser(description=desc)
    parser.add_argument('mode',
                        help='Mode to run the program in. Can be one of [c, '
                             't] (classification or training)',
                        choices=['t', 'c'])
    parser.add_argument('-n', '--network', type=str,
                        help='Path to the file containing a trained model. '
                             'Only applies to classification mode.')
    parser.add_argument('-i', '--image', type=str,
                        help='Path to the file containing a trained model. '
                             'Only applies to classification mode.')
    parser.add_argument('-e', '--epochs', type=int,
                        help='The number of epochs to train the neural '
                             'network. Only applies to training mode.')
    parser.add_argument('--name', type=str,
                        help='The path to save the file containing the trained'
                             'model. Only applies to training mode')
    return parser.parse_args()


def main(args):
    if args.mode == 't':
        train(args.epochs, args.name)
    else:
        classify(args.image, args.network)


if __name__ == '__main__':
    main(parse_arguments())

# Suggested commands for running the script:
# ----- Training -----
# python classifier.py t -e 10

# ----- Classifying -----
# python classifier.py c -i dog.jpg -n network_0.m5
