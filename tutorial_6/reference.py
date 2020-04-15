CLASS_LABELS = ['plane', 'car', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck']


def main():
    name = "cifar10.h5"
    (x_tr, y_tr), (x_tst, y_tst) = cifar10.load_data()
    print('x_tr shape = {}'.format(x_tr.shape))
    print('y_tr shape = {}'.format(y_tr.shape))

    image = random.randint(0,x_tr.shape[0])
    image_class = y_tr[image][0]
    print('Image class --> {}'.format(CLASS_LABELS[image_class]))
    cv.imshow('image', x_tr[image])
    cv.waitKey(0)

    y_tr = to_categorical(y_tr, len(CLASS_LABELS))
    y_tst = to_categorical(y_tst, len(CLASS_LABELS))

    print('One hot encoding label = {}'.format(y_tr[image]))


    # FEATURE EXTRACTION
    first_convolution = Conv2D(32, (3,3), activation='relu', padding='same',
                               name='first_convolution', input_shape=(32,32,3))
    second_convolution = Conv2D(32, (3,3), activation='relu', padding='same',
                               name='second_convolution')
    scaling_down = MaxPooling2D(pool_size=(2,2))

    # CLASSIFICATION
    classifier_input = Flatten(name='classifier_input')
    classifier_hidden = Dense(512, activation='relu', name='classifier_hidden')
    overfitting_countermeasure = Dropout(0.5, name='overfitting_countermeasure')
    classifier_output = Dense(len(CLASS_LABELS), activation='softmax', name='classifier_output')

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

    network.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
    network.fit(x_tr, y_tr, batch_size=32, epochs=1, validation_split=0.2)
    network.save(name)
    print(f"network saved to file: {name}")

def load_model(path):
    model = load_model(path)

def parse_arguments():
    parser = ArgumentParser()
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    main()
