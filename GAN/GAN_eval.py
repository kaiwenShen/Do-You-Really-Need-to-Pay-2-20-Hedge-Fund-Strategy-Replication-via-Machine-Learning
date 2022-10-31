import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.linalg import sqrtm
from scipy.stats import kstest, wasserstein_distance
from sklearn import metrics
from scipy.special import rel_entr
from sklearn.metrics import r2_score
from sklearn.naive_bayes import GaussianNB
from statsmodels.distributions import ECDF
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.stattools import acf


class GAN_eval():
    def __init__(self, real, fake, dataset, subplot_title, model_name):
        assert isinstance(real, np.ndarray)
        assert isinstance(fake, np.ndarray)
        assert isinstance(dataset, np.ndarray)
        assert isinstance(subplot_title, list)
        assert isinstance(model_name, list)
        assert real.ndim == fake.ndim
        self.real = real
        self.fake = fake
        self.dataset = dataset
        self.subplot_title = subplot_title
        self.model_name = model_name

    # from Pros and Cons of GAN Evaluation Measures
    def FID(self, real=None, fake=None, dataset=None):
        '''
            Fréchet Inception Distance (FID)
            improvement over the existing inception score, more robust to noise than IS
            lower, better
        :return:
        '''
        if real is None:
            real = self.real
        if fake is None:
            fake = self.fake
        if dataset is None:
            dataset = self.dataset
        assert real.shape == fake.shape
        if real.ndim > 2 or fake.ndim > 2:
            # we eliminate first dimension
            real = real.reshape(real.shape[0] * real.shape[1], real.shape[2])
            fake = fake.reshape(fake.shape[0] * fake.shape[1], fake.shape[2])

        # calculate mean and covariance statistics
        mu1, sigma1 = real.mean(axis=0), np.cov(real, rowvar=False)
        mu2, sigma2 = fake.mean(axis=0), np.cov(fake, rowvar=False)
        # calculate sum squared difference between means
        ssdiff = np.sum((mu1 - mu2) ** 2.0)
        # calculate sqrt of product between cov
        covmean = sqrtm(sigma1.dot(sigma2))
        # check and correct imaginary numbers from sqrt
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        # calculate score
        fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        return fid

    def linear_MMD(self, real=None, fake=None, dataset=None):
        '''
            Maximum Mean Discrepancy (MMD) using linear kernels
            lower,better
        :return:
        '''
        if real is None:
            real = self.real
        if fake is None:
            fake = self.fake
        if dataset is None:
            dataset = self.dataset
        assert real.shape == fake.shape
        if real.ndim > 2 or fake.ndim > 2:
            # we eliminate first dimension using average, no concatination because memory demanding.(4.29G)
            real = np.mean(real, axis=0)
            fake = np.mean(fake, axis=0)
        XX = np.dot(real, real.T).mean()
        YY = np.dot(fake, fake.T).mean()
        XY = np.dot(real, fake.T).mean()
        return XX + YY - 2 * XY

    def gaussian_MMD(self, real=None, fake=None, dataset=None, gamma=1.0):
        """MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))
        Arguments:
            X {[n_sample1, dim]} -- [X matrix]
            Y {[n_sample2, dim]} -- [Y matrix]
        Keyword Arguments:
            gamma {float} -- [kernel parameter] (default: {1.0})
        Returns:
            [scalar] -- [MMD value]
        """
        if real is None:
            real = self.real
        if fake is None:
            fake = self.fake
        if dataset is None:
            dataset = self.dataset
        assert real.shape == fake.shape
        if real.ndim > 2 or fake.ndim > 2:
            # we eliminate first dimension using average, no concatination because memory demanding.(4.29G)
            real = np.mean(real, axis=0)
            fake = np.mean(fake, axis=0)
        XX = metrics.pairwise.rbf_kernel(real, real, gamma)
        YY = metrics.pairwise.rbf_kernel(fake, fake, gamma)
        XY = metrics.pairwise.rbf_kernel(real, fake, gamma)
        return XX.mean() + YY.mean() - 2 * XY.mean()

    def poly_MMD(self, real=None, fake=None, dataset=None, degree=2, gamma=1, coef0=0):
        """MMD using polynomial kernel (i.e., k(x,y) = (gamma <X, Y> + coef0)^degree)
        Arguments:
            X {[n_sample1, dim]} -- [X matrix]
            Y {[n_sample2, dim]} -- [Y matrix]
        Keyword Arguments:
            degree {int} -- [degree] (default: {2})
            gamma {int} -- [gamma] (default: {1})
            coef0 {int} -- [constant item] (default: {0})
        Returns:
            [scalar] -- [MMD value]
        """
        if real is None:
            real = self.real
        if fake is None:
            fake = self.fake
        if dataset is None:
            dataset = self.dataset
        assert real.shape == fake.shape
        if real.ndim > 2 or fake.ndim > 2:
            # we eliminate first dimension using average, no concatination because memory demanding.(4.29G)
            real = np.mean(real, axis=0)
            fake = np.mean(fake, axis=0)
        XX = metrics.pairwise.polynomial_kernel(real, real, degree, gamma, coef0)
        YY = metrics.pairwise.polynomial_kernel(fake, fake, degree, gamma, coef0)
        XY = metrics.pairwise.polynomial_kernel(real, fake, degree, gamma, coef0)
        return XX.mean() + YY.mean() - 2 * XY.mean()

    def kl_div(self, real=None, fake=None, dataset=None, div_only=True):
        """
        Kullback-Leibler Divergence
        :return: kl divergence (in bits) and kl distance
        """
        if real is None:
            real = self.real
        if fake is None:
            fake = self.fake
        if dataset is None:
            dataset = self.dataset
        # first build a naive bayes classifier
        assert real.ndim == 2 or real.ndim == 3
        assert dataset.ndim == 3
        assert real.shape == fake.shape

        # reshape dataset
        Tdataset = []
        Treal = []
        Tfake = []
        for i in range(dataset.shape[0]):
            Tdataset.append(dataset[i].T)
        if real.ndim > 2:
            for i in range(real.shape[0]):
                Treal.append(real[i].T)
                Tfake.append(fake[i].T)
            Treal = np.array(Treal)
            Tfake = np.array(Tfake)
            Treal = Treal.reshape(Treal.shape[0] * Treal.shape[1], Treal.shape[2])
            Tfake = Tfake.reshape(Tfake.shape[0] * Tfake.shape[1], Tfake.shape[2])

        else:
            Treal = real.T
            Tfake = fake.T

        Tdataset = np.array(Tdataset)

        Tdataset = Tdataset.reshape(Tdataset.shape[0] * Tdataset.shape[1], Tdataset.shape[2])

        gbn = GaussianNB()
        gbn.fit(
            Tdataset,
            np.repeat(np.arange(real.shape[2]), dataset.shape[0])
        )
        real_lprob = gbn.predict_proba(Treal)
        fake_lprob = gbn.predict_proba(Tfake)
        res = []
        for i in range(real_lprob.shape[0]):
            res.append(sum(rel_entr(fake_lprob[i], real_lprob[i])))
        if div_only:
            return np.mean(res)
        else:
            return np.mean(res), np.mean(np.sqrt(res).tolist())

    def js_div(self, real=None, fake=None, dataset=None, div_only=True):
        """
        Jensen-Shannon Divergence
        :return: js divergence in bits, js distance
        """
        if real is None:
            real = self.real
        if fake is None:
            fake = self.fake
        if dataset is None:
            dataset = self.dataset
        # first build a naive bayes classifier
        assert real.ndim == 2 or real.ndim == 3
        assert dataset.ndim == 3
        assert real.shape == fake.shape

        # reshape dataset
        Tdataset = []
        Treal = []
        Tfake = []
        for i in range(dataset.shape[0]):
            Tdataset.append(dataset[i].T)
        if real.ndim > 2:
            for i in range(real.shape[0]):
                Treal.append(real[i].T)
                Tfake.append(fake[i].T)
            Treal = np.array(Treal)
            Tfake = np.array(Tfake)
            Treal = Treal.reshape(Treal.shape[0] * Treal.shape[1], Treal.shape[2])
            Tfake = Tfake.reshape(Tfake.shape[0] * Tfake.shape[1], Tfake.shape[2])

        else:
            Treal = real.T
            Tfake = fake.T

        Tdataset = np.array(Tdataset)

        Tdataset = Tdataset.reshape(Tdataset.shape[0] * Tdataset.shape[1], Tdataset.shape[2])

        gbn = GaussianNB()
        gbn.fit(
            Tdataset,
            np.repeat(np.arange(real.shape[2]), dataset.shape[0])
        )
        real_lprob = gbn.predict_proba(Treal)
        fake_lprob = gbn.predict_proba(Tfake)
        res = []
        for i in range(real_lprob.shape[0]):
            m = 0.5 * (fake_lprob[i] + real_lprob[i])
            res.append(0.5 * sum(rel_entr(fake_lprob[i], m)) + 0.5 * sum(rel_entr(real_lprob[i], m)))
        if div_only:
            return np.mean(res)
        else:
            return np.mean(res), np.mean(np.sqrt(res).tolist())

    def Inception_score(self, real=None, fake=None, dataset=None):
        '''

        :param real:
        :param fake:
        :param dataset:
        :return: 1 means real==fake, higher the output, the more different between fake and real
        '''
        if real is None:
            real = self.real
        if fake is None:
            fake = self.fake
        if dataset is None:
            dataset = self.dataset
        kld, _ = self.kl_div(real, fake, dataset, div_only=False)
        return np.exp(np.mean(kld))

    # from sigwgan

    def ks_test(self, real=None, fake=None, dataset=None, group=True, p_val_only=True):
        if real is None:
            real = self.real
        if fake is None:
            fake = self.fake
        if dataset is None:
            dataset = self.dataset
        assert real.shape == fake.shape
        assert real.ndim == 2 or real.ndim == 3
        if real.ndim == 3:
            real = real.reshape(real.shape[0] * real.shape[1], real.shape[2])
            fake = fake.reshape(fake.shape[0] * fake.shape[1], fake.shape[2])
        res = []
        for i in range(real.shape[1]):
            stats, pval = kstest(real[:, i], fake[:, i])
            res.append([stats, pval])
        if group:
            if p_val_only:
                return np.mean(res, axis=0)[1]
            return np.mean(res, axis=0)
        else:
            return pd.DataFrame(res)

    def lp_dist(self, real=None, fake=None, dataset=None, ord=2, group=True):
        if real is None:
            real = self.real
        if fake is None:
            fake = self.fake
        if dataset is None:
            dataset = self.dataset
        assert real.shape == fake.shape
        assert real.ndim == 2 or real.ndim == 3
        if real.ndim == 3:
            real = real.reshape(real.shape[0] * real.shape[1], real.shape[2])
            fake = fake.reshape(fake.shape[0] * fake.shape[1], fake.shape[2])
        res = []
        for i in range(real.shape[1]):
            res.append(np.linalg.norm(real[:, i] - fake[:, i], ord=ord) / real.shape[0])  # adjust for num obs
        if group:
            return np.mean(res)
        return res

    def wasserstein(self, real=None, fake=None, dataset=None, group=True):
        if real is None:
            real = self.real
        if fake is None:
            fake = self.fake
        if dataset is None:
            dataset = self.dataset
        assert real.shape == fake.shape
        assert real.ndim == 2 or real.ndim == 3
        if real.ndim == 3:
            real = real.reshape(real.shape[0] * real.shape[1], real.shape[2])
            fake = fake.reshape(fake.shape[0] * fake.shape[1], fake.shape[2])
        res = []
        for i in range(real.shape[1]):
            res.append(wasserstein_distance(real[:, i], fake[:, i]))
        if group:
            return np.mean(res)
        return res

    def ACF(self, real=None, fake=None, dataset=None, nlags=17, group=True):
        '''
        absolute error of the autocorrelation function score.
        :return:
        '''
        if real is None:
            real = self.real
        if fake is None:
            fake = self.fake
        if dataset is None:
            dataset = self.dataset
        assert real.shape == fake.shape
        assert real.ndim == 2 or real.ndim == 3
        if real.ndim == 3:
            real_acf = []
            fake_acf = []
            for i in range(real.shape[0]):
                # because we assert real and fake the same shape
                real_acf_curr = []
                fake_acf_curr = []
                for j in range(real.shape[2]):
                    real_acf_curr.append(acf(real[i][:, j], nlags=nlags))
                    fake_acf_curr.append(acf(fake[i][:, j], nlags=nlags))
                real_acf.append(real_acf_curr)
                fake_acf.append(fake_acf_curr)
            # real_acf=np.array(real_acf)
            # fake_acf=np.array(fake_acf)
            real_acf = np.mean(real_acf, axis=0)
            fake_acf = np.mean(fake_acf, axis=0)
            res = []
            for i in range(real_acf.shape[1]):
                res.append(np.mean(abs(real_acf[i] - fake_acf[i])))
            if group:
                return np.mean(res)
            return res
        else:
            res = []
            for i in range(real.shape[1]):
                res.append(np.mean(abs(acf(real[:, i], nlags=nlags) - acf(fake[:, i], nlags=nlags))))
            if group:
                return np.mean(res)
            return res

    def R2_relative_error(self, real=None, fake=None, dataset=None, group=True):
        '''
        split whole data into real and dataset
        train on dataset, predict real and fake
        assume all dates are in accending order
        :return:
        '''
        if real is None:
            real = self.real
        if fake is None:
            fake = self.fake
        if dataset is None:
            dataset = self.dataset
        assert dataset.ndim == 3
        assert real.ndim == 3
        assert fake.ndim == 3
        col_name = [f'A{i}' for i in range(dataset.shape[2])]

        def dfxy(df, col):
            df = df.reshape(df.shape[0] * df.shape[1], df.shape[2])
            df = pd.DataFrame(df, columns=col_name)
            return df.shift(-1).dropna()[col], df.iloc[:-1, :].loc[:, df.columns != col]

        res = []
        for col in col_name:
            Y_train, X_train = dfxy(dataset, col)
            Y_test, X_test = dfxy(real, col)
            Y_interpo, X_interpo = dfxy(real, col)
            ols_model = OLS(Y_train, X_train).fit()
            Y_pred = ols_model.predict(X_test)
            Y_interpo_pred = ols_model.predict(X_interpo)
            res.append(abs(r2_score(Y_test, Y_pred) - r2_score(Y_interpo, Y_interpo_pred)))
        if group:
            return np.mean(res)
        return res

    def eyeball(self, real=None, fake=None, dataset=None, subplot_title=None):
        '''
        eyeball the histogram distribution of each hedge fund return
        :return:
        '''
        if real is None:
            real = self.real
        if fake is None:
            fake = self.fake
        if dataset is None:
            dataset = self.dataset
        if subplot_title is None:
            subplot_title = self.subplot_title
        assert real.ndim == 3
        assert fake.ndim == 3
        if not isinstance(subplot_title, list):
            raise TypeError
        assert len(subplot_title) == real.shape[2]
        real = real.reshape(real.shape[0] * real.shape[1], real.shape[2])
        fake = fake.reshape(fake.shape[0] * fake.shape[1], fake.shape[2])
        row, col = 0, 0
        fig, ax = plt.subplots(12, 3, figsize=(20, 30))
        for i in range(real.shape[1]):
            ecdf_real = ECDF(real[:, i])
            ecdf_fake = ECDF(fake[:, i])
            x = np.linspace(min(real[:, i]), max(real[:, i]))
            real_cdf = ecdf_real(x)
            generated_cdf = ecdf_fake(x)
            ax[row, col].step(x, real_cdf)
            ax[row, col].step(x, generated_cdf)
            ax[row, col].set_title(subplot_title[i])
            ax[row, col].legend(['True', 'Generated'], loc="upper left")
            col += 1
            if col == 3:
                col = 0
                row += 1
        plt.suptitle(self.model_name[0],y=1,fontsize=24)
        fig.tight_layout()
        plt.show()

    def run_all(self):
        res = []
        metrics = []
        for i, name in enumerate(dir(self)):
            obj = getattr(self, name)
            if callable(obj) and name != 'run_all' and name != 'eyeball' and name[:2] != '__':
                data = obj()
                res.append(data)
                metrics.append(name)
                print(f"{i + 1} out of {len(dir(self))} done.")
        self.eyeball()
        return pd.DataFrame(res, index=metrics, columns=self.model_name)


if __name__ == '__main__':
    import pickle


    def dic_read(loc):
        a_file = open(loc, "rb")
        output = pickle.load(a_file)
        return output


    hfd_fullname = dic_read('../cleaned_data/hfd_fullname.pkl')
    factor_etf_name = dic_read('../cleaned_data/factor_etf_name.pkl')

    all_data_name = {**factor_etf_name, **hfd_fullname}

    real = np.random.normal(size=(500, 48, 35))
    fake = np.random.normal(size=(500, 48, 35))
    dataset = np.random.normal(size=(500, 48, 35))
    evaluate = GAN_eval(real, fake, dataset, list(all_data_name.values()), ['Benchmark'])
    # res = evaluate.run_all()
    # print(res)
    evaluate.eyeball()