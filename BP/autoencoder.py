from keras.datasets import mnist
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.callbacks import TensorBoard
import numpy as np
from keras import backend as K
import matplotlib.pyplot as plt

(x_train, _), (x_test, _) = mnist.load_data()

input = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format


def encoder(x):
    conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    maxpool1 = MaxPooling2D((2, 2), padding='same')(conv1)
    conv2 = Conv2D(8, (3, 3), activation='relu', padding='same')(maxpool1)
    maxpool2 = MaxPooling2D((2, 2), padding='same')(conv2)
    conv3 = Conv2D(2, (3, 3), activation='relu', padding='same')(maxpool2)
    maxpool3 = MaxPooling2D((2, 2), padding='same', name= "encoder")(conv3)
    return maxpool3


def decoder(x):
    resize1 = UpSampling2D((2, 2))(x)
    conv1 = Conv2D(8, (3, 3), activation='relu', padding='same')(resize1)
    resize2 = UpSampling2D((2, 2))(conv1)
    conv2 = Conv2D(16, (3, 3), activation='relu')(resize2)
    resize3 = UpSampling2D((2, 2))(conv2)
    conv3 = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(resize3)
    return conv3

autoencoder = Model(input, decoder(encoder(input)))
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

encoder = Model(autoencoder.input, autoencoder.get_layer("encoder").output)

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format

autoencoder.fit(x_train, x_train,
                epochs=5,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test))

decoded_imgs = autoencoder.predict(x_test)

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i + n + 1)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
