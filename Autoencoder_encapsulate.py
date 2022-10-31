"""
    encapsulated class AE
"""
import numpy as np
import pandas as pd
from keras import Model, Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense, LeakyReLU
from keras.losses import MeanSquaredError
from keras.optimizers import Nadam
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.regression.linear_model import OLS

from helper import normalization, price_impact, transaction_cost, reshape_cab, ex_post_return


class Autoencoder(Model):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = Sequential([
            Dense(latent_dim, input_dim=22, use_bias=False),
            LeakyReLU(alpha=.2)
        ])
        self.decoder = Sequential([
            Dense(22, input_dim=latent_dim, use_bias=False),
            LeakyReLU(alpha=.2)
        ])

    def call(self, x, **kwargs):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class AE:
    def __init__(self, x_train, y_train, x_test, y_test, latent_dim):
        '''
            data needs to be unscaled.
        :param x_train:
        :param y_train:
        :param x_test:
        :param y_test:
        :param latent_dim:
        '''
        self.reshape_strat_weight_on_etf = None
        self.window = None
        self.strat_weight_on_etf = None
        self.hfd = None
        self.rf = None
        self.hfd_fullname = None
        self._ante = None
        self.OOS_hfd = None
        self.OOS_etf = None
        self.OOS_rf = None
        self.test_scale = None
        assert len(x_train) == len(y_train) and len(y_test) == len(x_test)
        self.history = None
        self.autoencoder = None
        self.train_scale = MinMaxScaler()
        # self.test_scale = MinMaxScaler()

        self._x_train = self.train_scale.fit_transform(x_train)
        # self._x_test = self.test_scale.fit_transform(x_test)
        self._x_test = x_test
        self._y_train = y_train
        self._y_test = y_test
        self._latent_dim = latent_dim

    def train(self, patience=5, verbose=2, plot=True):
        '''
        AE training will only use self._x_train

        :return: plot
        '''
        self.autoencoder = Autoencoder(self._latent_dim)
        self.autoencoder.compile(
            optimizer=Nadam(),
            loss=MeanSquaredError()
        )
        self.history = self.autoencoder.fit(
            self._x_train,
            self._x_train,
            epochs=1000,
            verbose=verbose,
            batch_size=48,
            validation_split=.25,
            callbacks=[EarlyStopping(
                monitor='val_loss',
                patience=patience,
                mode='auto'
            )
            ]
        )
        if plot:
            print(self.autoencoder.summary())
            plt.plot(self.history.history['loss'])
            plt.plot(self.history.history['val_loss'])
            plt.title('Model Loss')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'val'], loc='upper left')
            plt.show()

    def model_IS_r2(self):
        x_pred = self.autoencoder.predict(self._x_train, verbose=0)
        return r2_score(self._x_train, x_pred)

    def model_IS_RMSE(self):
        x_pred = self.autoencoder.predict(self._x_train, verbose=0)
        return mean_squared_error(self._x_train, x_pred, squared=False)

    def model_OOS_r2(self):
        seq = []
        for i in range(2, len(self._x_test)):
            scaler = MinMaxScaler()
            x_real = scaler.fit_transform(self._x_test[:i])
            x_pred = self.autoencoder.predict(x_real, verbose=0)
            seq.append(r2_score(x_real, x_pred))
        return seq

    def model_OOS_RMSE(self):
        seq = []
        for i in range(2, len(self._x_test)):
            scaler = MinMaxScaler()
            x_real = scaler.fit_transform(self._x_test[:i])
            x_pred = self.autoencoder.predict(x_real, verbose=0)
            seq.append(mean_squared_error(x_real, x_pred, squared=False))
        return seq

    def ante(self, rf, hfd, window=24, ):
        '''
        calculate ex-ante and ex-post return
        :return: ex-ante, ex-post
        '''
        assert isinstance(rf, pd.DataFrame)
        # extract main factor
        main_factor = self.autoencoder.encoder.predict(self._x_test, verbose=0)
        # OLS beta calculation

        # rolling window OLS window=24, consistent with the benchmark
        window = window
        start, end = 0, window
        ae_ols_beta = []
        normalization_factor = []
        for i in range(len(self._x_test) - window):
            X = main_factor[start:end]
            Y = self._y_test[start: end]
            beta = OLS(Y, X).fit().params
            ae_ols_beta.append(beta)
            # still need normalization factor.
            normalization_factor.append(normalization(Y, X, beta, window))
            start += 1
            end += 1

        # extract real weights on ETF
        factor_weight_on_etf = self.autoencoder.decoder.get_weights()[0]
        strat_weight_on_etf = []
        delta_weight = []
        for i in range(len(ae_ols_beta)):
            leakyrelu_weight = np.ones(factor_weight_on_etf.shape[1])
            for idx, val in enumerate(main_factor[window + i] @ factor_weight_on_etf):
                if val < 0:
                    leakyrelu_weight[idx] = 0.2
            strat_weight = (ae_ols_beta[0].T @ factor_weight_on_etf * leakyrelu_weight).T * normalization_factor[0]
            delta_weight.append(1 - np.sum(strat_weight, axis=0))
            strat_weight_on_etf.append(strat_weight)

        '''
        we are using insample ols to predict next step weighting on etf.
        insample: 0-12, generate ols beta,
        predict: out-sample lambda = in-sample beta,
        predict_return: t=13, rf * (1-sum(lambda))+lambda * etf_return
        therefore the last window in variable strat_weight_on_etf is invalid (no corresponding etf)
        '''
        # remove last element of weight and pop
        strat_weight_on_etf.pop()
        delta_weight.pop()
        # OOS ETF is x_test, OOS hfd is y_test

        self.OOS_etf = np.array(self._x_test.iloc[-len(strat_weight_on_etf):])
        self.OOS_hfd = self._y_test.iloc[-len(strat_weight_on_etf):]
        self.OOS_rf = np.array(rf.iloc[-len(strat_weight_on_etf):])
        # calculate ante return
        ae_ret_ante = []
        for idx, strat_weight in enumerate(strat_weight_on_etf):
            ret_ante = delta_weight[idx] * self.OOS_rf[idx] + np.sum(self.OOS_etf[idx] * strat_weight.T, axis=1)
            ae_ret_ante.append(ret_ante)
        ae_ret_ante = pd.DataFrame(ae_ret_ante)
        ae_ret_ante.columns = hfd.columns
        ae_ret_ante.index = hfd.index[-len(ae_ret_ante):]
        # capture result
        self._ante = ae_ret_ante
        self.rf = rf
        self.hfd = hfd
        self.strat_weight_on_etf = strat_weight_on_etf
        self.reshape_strat_weight_on_etf = reshape_cab(strat_weight_on_etf)
        self.window = window
        return self._ante

    def post(self,factor_etf_data):
        if self._ante is None:
            raise Exception('please execute ante before turnover')
        OOS_factor_etf = (factor_etf_data.iloc[-len(self.reshape_strat_weight_on_etf[0]) - self.window:])  # include the first window
        self._post = ex_post_return(self._ante,self.window,self.reshape_strat_weight_on_etf,OOS_factor_etf)
        return self._post

    def turnover(self, hfd_fullname):
        if self._ante is None:
            raise Exception('please execute ante before turnover')
        turnover = np.zeros(len(self.hfd.columns))
        for i in range(len(self.strat_weight_on_etf) - 1):
            turnover += np.sum(abs(self.strat_weight_on_etf[i] - self.strat_weight_on_etf[1 + i]), axis=0)
        turnover /= len(self.strat_weight_on_etf)/12
        turnover_df = []
        for i in range(len(turnover)):
            turnover_df.append([list(hfd_fullname.values())[i], turnover[i]])
        turnover_df = pd.DataFrame(turnover_df, columns=['Real_AE', 'Turnover'])
        turnover_df = turnover_df.set_index('Real_AE')
        # assign input
        self.hfd_fullname = hfd_fullname
        return turnover_df

    def plot(self,hfd_fullname,title=None):
        assert isinstance(title, str)
        fig, ax = plt.subplots(5, 3, figsize=(30, 20))
        row, col = 0, 0
        for idx, strat in enumerate(self._ante.columns):
            temp = pd.DataFrame(
                [self._ante.iloc[:, idx].cumsum(), self._post.iloc[:, idx].cumsum(), self.OOS_hfd.iloc[:, idx].cumsum()],
                index=['Ex-ante', 'Ex_post', 'Real']).T
            for i, name in enumerate(temp.columns):
                ax[row][col].plot(temp.iloc[:, i], label=name)
                ax[row][col].legend(loc="upper left")
            ax[row][col].set_title(hfd_fullname[strat])
            col += 1
            if col % 3 == 0:
                row += 1
                col = 0
        plt.suptitle(title, y=0.93, fontsize=24)
        plt.show()
