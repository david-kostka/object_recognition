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
    
    objectness_out = Dense(1, activation='sigmoid')(x)
    #objectness_out = Dense(1)(x)
    #objectness_out = Softmax()(objectness_out)
    output = concatenate([bbox_out, objectness_out])

    model = Model(inputs=input_image, outputs=output)

    return model

def decode_output(y_true, y_pred):
    # Cast
    y_true = K.cast(y_true, dtype=tf.float32)
    y_pred = K.cast(y_pred, dtype=tf.float32)

    # Decode Output
    conf_true = y_true[..., 4]
    conf_pred = y_pred[..., 4]
    bboxes_true = y_true[..., :4] 
    bboxes_pred = y_pred[..., :4]
    return conf_true, conf_pred, bboxes_true, bboxes_pred

def bbox_mse(y_true, y_pred):
    conf_true, conf_pred, bboxes_true, bboxes_pred = decode_output(y_true, y_pred)

    # Mask based on true confidence
    bboxes_true = bboxes_true * K.expand_dims(conf_true)
    bboxes_pred = bboxes_pred * K.expand_dims(conf_true)

    # Calculate Loss
    mseFunc = losses.MeanSquaredError(reduction='sum_over_batch_size')
    mse = mseFunc(bboxes_true, bboxes_pred)

    #print(mse)
    return mse

def conf_bce(y_true, y_pred):
    conf_true, conf_pred, _, _ = decode_output(y_true, y_pred)

    bceFunc = losses.BinaryCrossentropy(reduction='sum_over_batch_size')
    bce = bceFunc(conf_true, conf_pred)

    #print(bce)
    return bce

def nao_loss(y_true, y_pred):
    mse = bbox_mse(y_true, y_pred)
    bce = conf_bce(y_true, y_pred)
    return mse + bce

def test_loss():
    # Loss Funktion und Metriken testen
    pred_test = np.array([[1.0, 1.0, 1.0, 1.0, 1], [0.7, 0.8, 0.7, 0.7, 1]])
    true_test = np.array([[1.0, 1.0, 1.0, 1.0, 1], [0.3, 0.3, 0.4, 0.4, 1]])
    print(true_test)
    print(pred_test)
    print('-------- Loss ----------')
    print(md.nao_loss(true_test, pred_test))