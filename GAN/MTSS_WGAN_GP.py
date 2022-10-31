# Large amount of credit goes to:
# https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py
# which I've used as a reference for this implementation

from __future__ import print_function, division

import pickle
from datetime import datetime
from random import randint
import tensorflow as tf
from keras.datasets import mnist
from keras.layers.merging.base_merge import _Merge
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, LSTM, LayerNormalization
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
from functools import partial

import keras.backend as K

import matplotlib.pyplot as plt

import sys

import numpy as np

from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()

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


# class RandomWeightedAverage(_Merge):
#     """
#     Provides a (random) weighted average between real and generated image samples
#     Warning: the first dimension of the random_uniform needs to be the same as the batchsize
#     """
#
#     def _merge_function(self, inputs):
#         alpha = K.random_uniform((32, 1, 1))
#         return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])


class WGAN_GP():
    def __init__(self, dataset):
        self.X_train = dataset

        self.ts_length = self.X_train.shape[1]
        self.ts_feature = self.X_train.shape[2]

        self.ts_shape = (self.ts_length, self.ts_feature)

        self.latent_shape = (self.ts_length, self.ts_feature)

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
        optimizer = RMSprop(learning_rate=0.00005)

        # Build the generator and critic
        self.generator = self.build_generator()
        self.critic = self.build_critic()

        # -------------------------------
        # Construct Computational Graph
        #       for the Critic
        # -------------------------------

        # Freeze generator's layers while training critic
        self.generator.trainable = False

        # Image input (real sample)
        real_ts = Input(shape=self.ts_shape)

        # Noise input
        z_disc = Input(shape=(self.ts_length, self.ts_feature,))
        # Generate image based of noise (fake sample)
        fake_ts = self.generator(z_disc)

        # Discriminator determines validity of the real and fake images
        fake = self.critic(fake_ts)
        valid = self.critic(real_ts)

        # Construct weighted average between real and fake images
        interpolated_img = self.RandomWeightedAverage()([real_ts, fake_ts])
        # Determine validity of weighted sample
        validity_interpolated = self.critic(interpolated_img)

        # Use Python partial to provide loss function with additional
        # 'averaged_samples' argument
        partial_gp_loss = partial(self.gradient_penalty_loss,
                                  averaged_samples=interpolated_img)
        partial_gp_loss.__name__ = 'gradient_penalty'  # Keras requires function names

        self.critic_model = Model(inputs=[real_ts, z_disc],
                                  outputs=[valid, fake, validity_interpolated])
        self.critic_model.compile(loss=[self.wasserstein_loss,
                                        self.wasserstein_loss,
                                        partial_gp_loss],
                                  optimizer=optimizer,
                                  loss_weights=[1, 1, 10])
        # -------------------------------
        # Construct Computational Graph
        #         for Generator
        # -------------------------------

        # For the generator we freeze the critic's layers
        self.critic.trainable = False
        self.generator.trainable = True

        # Sampled noise for input to generator
        z_gen = Input(shape=(self.ts_length, self.ts_feature,))
        # Generate images based of noise
        ts = self.generator(z_gen)
        # Discriminator determines validity
        valid = self.critic(ts)
        # Defines generator model
        self.generator_model = Model(z_gen, valid)
        self.generator_model.compile(loss=self.wasserstein_loss, optimizer=optimizer)

    class RandomWeightedAverage(_Merge):
        """
        Provides a (random) weighted average between real and generated image samples
        Warning: the first dimension of the random_uniform needs to be the same as the batchsize
        """

        def _merge_function(self, inputs):
            alpha = K.random_uniform((32, 1, 1))
            return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
        """
        Computes gradient penalty based on prediction and weighted real / fake samples
        """
        gradients = K.gradients(y_pred, averaged_samples)[0]
        # compute the euclidean norm by squaring ...
        gradients_sqr = K.square(gradients)
        #   ... summing over the rows ...
        gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))
        #   ... and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = K.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples
        return K.mean(gradient_penalty)

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def build_generator(self):
        model = Sequential(
            [
                LSTM(100, input_shape=self.latent_shape, activation='sigmoid', return_sequences=True),
                LayerNormalization(),
                LSTM(100, return_sequences=True, activation='sigmoid'),
                LeakyReLU(alpha=.2),
                LayerNormalization(),
                Dense(self.ts_feature)
            ])
        model.summary()
        noise = Input(shape=(self.ts_length, self.ts_feature,))
        ts = model(noise)

        return Model(noise, ts)

    def build_critic(self):
        model = Sequential(
            [
                LSTM(100, input_shape=self.ts_shape, return_sequences=True),
                LSTM(100, return_sequences=True),
                Flatten(),
                Dense(1)
            ]
        )

        model.summary()

        ts = Input(shape=self.ts_shape)
        validity = model(ts)

        return Model(ts, validity)

    def train(self, epochs, batch_size, sample_interval=50):

        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake = np.ones((batch_size, 1))
        dummy = np.zeros((batch_size, 1))  # Dummy gt for gradient penalty
        for epoch in range(epochs):

            for _ in range(self.n_critic):
                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random batch of images
                idx = np.random.randint(0, self.X_train.shape[0], batch_size)
                imgs = self.X_train[idx]
                # Sample generator input
                noise = np.random.normal(0, 1, (batch_size, self.ts_length, self.ts_feature))
                # Train the critic
                d_loss = self.critic_model.train_on_batch(
                    [imgs, noise],
                    [valid, fake, dummy])

            # ---------------------
            #  Train Generator
            # ---------------------
            tf.compat.v1.experimental.output_all_intermediates(True)
            g_loss = self.generator_model.train_on_batch(noise, valid)

            # Plot the progress
            print("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss[0], g_loss))
        time_now = datetime.now().strftime("%Y%m%d_%H-%M-%S")
        self.generator.compile(optimizer=RMSprop(), loss='binary_crossentropy')
        self.generator.save(f'./trained_generator/MTSS_GAN_GP{time_now}.h5')


if __name__ == '__main__':
    wgan = WGAN_GP(dataset)
    wgan.train(epochs=5000, batch_size=32, sample_interval=100)
