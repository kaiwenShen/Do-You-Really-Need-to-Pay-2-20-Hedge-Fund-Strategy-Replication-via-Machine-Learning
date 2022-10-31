from __future__ import print_function, division

import pickle
from random import randint

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, LSTM, LayerNormalization
from keras.layers import BatchNormalization, LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt

import sys

import numpy as np
import pandas as pd

# Read-in cleaned data
from sklearn.preprocessing import MinMaxScaler


def read_csv(loc, date=True):
    df = pd.read_csv(loc)
    if date:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    return df


def dic_read(loc):
    a_file = open(loc, "rb")
    output = pickle.load(a_file)
    return output


def set_seed(seed_value=123):
    import os
    import random
    import tensorflow as tf
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    tf.random.set_seed(seed_value)
    from keras import backend as K
    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    tf.compat.v1.keras.backend.set_session(sess)
    K.set_session(sess)


def random_sampling(dataset, n_sample, window):
    '''
    implicitly assuming there is no calendar effect.
    :param dataset: np.ndarray
    :param n_sample:
    :param window:
    :return:
    '''
    isinstance(dataset, np.ndarray)
    step = 0
    res = []
    while step < n_sample:
        step += 1
        randidx = randint(0, dataset.shape[0] - window)
        res.append(dataset[randidx:window + randidx])
    # label as real data
    # label = np.ones(n_sample)
    # return np.array(res), label
    return np.array(res)


set_seed()

hfd = read_csv('../cleaned_data/hfd.csv')
factor_etf_data = read_csv('../cleaned_data/factor_etf_data.csv')
hfd_fullname = dic_read('../cleaned_data/hfd_fullname.pkl')
factor_etf_name = dic_read('../cleaned_data/factor_etf_name.pkl')

all_data_name = {**factor_etf_name, **hfd_fullname}

dataset = factor_etf_data.join(hfd)
data_scaler = MinMaxScaler()
data = data_scaler.fit_transform(dataset)

dataset = random_sampling(data, 1000, 48)


class GAN():
    def __init__(self, dataset):
        assert isinstance(dataset, np.ndarray)
        self.X_train = dataset

        self.ts_length = self.X_train.shape[1]
        self.ts_feature = self.X_train.shape[2]

        self.ts_shape = (self.ts_length, self.ts_feature)
        self.latent_shape = (self.ts_length, self.ts_feature)

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(
            loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates TS
        z = Input(shape=(self.ts_length, self.ts_feature,))
        ts = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated time series as input and determines validity
        validity = self.discriminator(ts)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):
        model = Sequential(
            [
                Dense(100, input_shape=self.latent_shape, activation='sigmoid', ),
                LeakyReLU(alpha=.2),
                LayerNormalization(),
                Dense(100, activation='sigmoid'),
                LeakyReLU(alpha=.2),
                LayerNormalization(),
                Dense(self.ts_feature)
            ])
        model.summary()
        noise = Input(shape=(self.ts_length, self.ts_feature,))
        ts = model(noise)

        return Model(noise, ts)

    def build_discriminator(self):
        model = Sequential(
            [
                Dense(100, input_shape=self.ts_shape, ),
                Dense(100, ),
                Dense(1, activation='sigmoid')
            ]
        )

        model.summary()

        ts = Input(shape=self.ts_shape)
        validity = model(ts)

        return Model(ts, validity)

    def train(self, epochs, batch_size=128, sample_interval=50):
        # Load the dataset
        # (X_train, _), (_, _) = mnist.load_data()
        #
        # # Rescale -1 to 1
        # X_train = X_train / 127.5 - 1.
        # X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of latent_shape
            idx = np.random.randint(0, self.X_train.shape[0], batch_size)
            ts = self.X_train[idx]

            noise = np.random.normal(0, 1, (batch_size, self.ts_length, self.ts_feature))

            # Generate a batch of new ts
            gen_ts = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(ts, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_ts, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, self.ts_length, self.ts_feature))

            # Train the generator (to have the discriminator label samples as valid)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))
        time_now = datetime.now().strftime("%Y%m%d_%H-%M-%S")
        self.generator.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy')
        self.generator.save(f'./trained_generator/GAN{time_now}.h5')


if __name__ == '__main__':
    from datetime import datetime

    gan = GAN(dataset)
    gan.train(epochs=5000, batch_size=32, sample_interval=200)
    # time_now = datetime.now().strftime("%Y%m%d_%H-%M-%S")
    # gan.generator.compile(optimizer=Adam(0.0002, 0.5),loss='binary_crossentropy')
    # gan.generator.save(f'./trained_generator/MTTS_GAN{time_now}.h5')
