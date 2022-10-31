from __future__ import print_function, division

import pickle
from datetime import datetime
from random import randint

import pandas as pd
from keras.layers import Input, Dense, LeakyReLU, LSTM, LayerNormalization
from keras.models import Sequential, Model
from keras.optimizers import RMSprop, Adam

import keras.backend as K

import numpy as np

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


class MTTS_WGAN():
    def __init__(self, dataset):
        isinstance(dataset, np.ndarray)
        self.X_train = dataset
        self.ts_length = self.X_train.shape[1]
        self.ts_feature = self.X_train.shape[2]

        self.ts_shape = (self.ts_length, self.ts_feature)
        self.latent_shape = (self.ts_length, self.ts_feature)

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
        self.clip_value = 0.01
        optimizer = RMSprop(learning_rate=0.00005)

        # Build and compile the critic
        self.critic = self.build_critic()
        self.critic.compile(loss=self.wasserstein_loss,
                            optimizer=optimizer,
                            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generated imgs
        z = Input(shape=(self.ts_length, self.ts_feature,))
        ts = self.generator(z)

        # For the combined model we will only train the generator
        self.critic.trainable = False

        # The critic takes generated images as input and determines validity
        valid = self.critic(ts)

        # The combined model  (stacked generator and critic)
        self.combined = Model(z, valid)
        self.combined.compile(loss=self.wasserstein_loss,
                              optimizer=optimizer,
                              metrics=['accuracy'])

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
        img = model(noise)

        return Model(noise, img)

    def build_critic(self):
        model = Sequential(
            [
                LSTM(100, input_shape=self.ts_shape, activation=None, return_sequences=True),
                LeakyReLU(alpha=0.2),
                LayerNormalization(),
                LSTM(100, return_sequences=True, activation=None),
                LeakyReLU(alpha=0.2),
                LayerNormalization(),
                Dense(1)  # we dont do sigmoid activation because critic output is supposed to be 1,-1
            ]
        )
        model.summary()

        ts = Input(shape=self.ts_shape)
        validity = model(ts)

        return Model(ts, validity)

    def train(self, epochs, batch_size=128, sample_interval=50):

        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake = np.ones((batch_size, 1))

        for epoch in range(epochs):

            for _ in range(self.n_critic):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random batch of images
                idx = np.random.randint(0, self.X_train.shape[0], batch_size)
                imgs = self.X_train[idx]

                # Sample noise as generator input
                noise = np.random.normal(0, 1, (batch_size, self.ts_length, self.ts_feature))

                # Generate a batch of new images
                gen_imgs = self.generator.predict(noise)

                # Train the critic
                d_loss_real = self.critic.train_on_batch(imgs, valid)
                d_loss_fake = self.critic.train_on_batch(gen_imgs, fake)
                d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

                # Clip critic weights
                for l in self.critic.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                    l.set_weights(weights)

            # ---------------------
            #  Train Generator
            # ---------------------

            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print("%d [D loss: %f] [G loss: %f]" % (epoch, 1 - d_loss[0], 1 - g_loss[0]))
        time_now = datetime.now().strftime("%Y%m%d_%H-%M-%S")
        self.generator.compile(optimizer=RMSprop(learning_rate=0.00005),
                               loss='binary_crossentropy')  # without compile, we cannot save model. Formality
        self.generator.save(f'./trained_generator/MTSS_WGAN{time_now}.h5')


if __name__ == '__main__':
    wgan = MTTS_WGAN(dataset)
    wgan.train(epochs=5000, batch_size=32, sample_interval=50)
