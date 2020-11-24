# Zweck:
# Definition eines Keras Models für Objekterkennung
# Author: David Kostka
# Datum: 05.11.2020

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, LeakyReLU, Reshape, Softmax, concatenate

def SingleNaoModel(img_size, bnorm=True):
    '''
    Liefert BBOX Koordinaten eines einzelnen Naos im Bild
    Basiert auf JET-Net von B-Human, aber leicht modifiziert und ohne 'depthwise seperable convolutions'
    Input: Image ((img_size), 1)
    Output: BBOX (4)

    Struktur:
    Bestehend aus 4 'Blöcken'
    Jeder Block besteht aus folgenden Layern:
        1. Batch Normalisierung
        2. n mal: Convolution (24 Kernels, 3x3) + Leaky ReLU
        3. Convolution mit strides=(2,2), emuliert Pooling Layer
        4. Leaky ReLU
    Letzter Block weicht ab: kein Pooling am Ende wie in den anderen Blöcken
    Output Layer ist ein Dense Layer mit 4 Nodes
    '''

    input_image = Input(shape=(*img_size, 1))

    if bnorm: 
        x = BatchNormalization()(input_image)
        x = Conv2D(24, (3, 3), strides=(1,1), padding='same', use_bias=False, kernel_initializer='he_uniform', dilation_rate=1)(x)
    else: 
        x = Conv2D(24, (3, 3), strides=(1,1), padding='same', use_bias=False, kernel_initializer='he_uniform', dilation_rate=1)(input_image)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(24, (3, 3), strides=(2,2), padding='same', use_bias=False, kernel_initializer='he_uniform', dilation_rate=1)(x)
    x = LeakyReLU(alpha=0.1)(x)

    if bnorm: x = BatchNormalization()(x)
    x = Conv2D(24, (3, 3), strides=(1,1), padding='same', use_bias=False, kernel_initializer='he_uniform', dilation_rate=1)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(24, (3, 3), strides=(1,1), padding='same', use_bias=False, kernel_initializer='he_uniform', dilation_rate=1)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(24, (3, 3), strides=(2,2), padding='same', use_bias=False, kernel_initializer='he_uniform', dilation_rate=1)(x)
    x = LeakyReLU(alpha=0.1)(x)

    if bnorm: x = BatchNormalization()(x)
    x = Conv2D(24, (3, 3), strides=(1,1), padding='same', use_bias=False, kernel_initializer='he_uniform', dilation_rate=1)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(24, (3, 3), strides=(1,1), padding='same', use_bias=False, kernel_initializer='he_uniform', dilation_rate=1)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(24, (3, 3), strides=(1,1), padding='same', use_bias=False, kernel_initializer='he_uniform', dilation_rate=1)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(24, (3, 3), strides=(2,2), padding='same', use_bias=False, kernel_initializer='he_uniform', dilation_rate=1)(x)
    x = LeakyReLU(alpha=0.1)(x)

    if bnorm: x = BatchNormalization()(x)
    x = Conv2D(24, (3, 3), strides=(1,1), padding='same', use_bias=False, kernel_initializer='he_uniform', dilation_rate=1)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(24, (3, 3), strides=(1,1), padding='same', use_bias=False, kernel_initializer='he_uniform', dilation_rate=1)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(24, (3, 3), strides=(1,1), padding='same', use_bias=False, kernel_initializer='he_uniform', dilation_rate=1)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(24, (3, 3), strides=(1,1), padding='same', use_bias=False, kernel_initializer='he_uniform', dilation_rate=1)(x)
    x = LeakyReLU(alpha=0.1)(x)

    #x = Conv2D(24, (1, 1), strides=(1,1), padding='same', use_bias=False, kernel_initializer='he_uniform', dilation_rate=1)(x)
    #x = LeakyReLU(alpha=0.1)(x)

    x = Flatten()(x)
    #output = Dense(4, activation="relu")(x)
    bbox_out = Dense(4)(x)
    bbox_out = LeakyReLU(alpha=0.1)(bbox_out)
    
    objectness_out = Dense(1)(x)
    objectness_out = Softmax()(objectness_out)

    output = concatenate([bbox_out, objectness_out])

    model = Model(inputs=input_image, outputs=output)

    return model

def TestModel():
    # Vielfaches von 4 für padding
    input_image = Input(shape=(16, 16, 1))

    #x = BatchNormalization()(input_image)
    x = Conv2D(16, (3, 3), strides=(1,1), padding='same', use_bias=False, kernel_initializer='he_uniform', dilation_rate=1)(input_image)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(16, (3, 3), strides=(2,2), padding='same', use_bias=False, kernel_initializer='he_uniform', dilation_rate=1)(x)
    x = LeakyReLU(alpha=0.1)(x)
    
    x = Flatten()(x)

    output = Dense(4)(x)

    model = Model(input_image, output)
    return model
