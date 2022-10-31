import pickle
from random import randint

import numpy as np
import pandas as pd
import os
import random
import tensorflow as tf

def normalization(Y, X, beta, window):
    Y = np.array(Y)
    X = np.array(X)
    beta = np.array(beta)
    R_hat = X @ beta
    denominator = np.sum((R_hat - np.mean(R_hat, axis=0)) ** 2 / (window - 1), axis=0)
    neumerator = np.sum((Y - np.mean(Y, axis=0)) ** 2 / (window - 1), axis=0)
    return np.sqrt(neumerator) / np.sqrt(denominator)
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


def transaction_cost(old_x,new_x,covMatrix,param=0.05):
    '''

    :param new_x:
    :param old_x:
    :param covMatrix: asset covariance
    :param param: transaction parameters
    :return:
    '''
    isinstance(param,float)
    covMatrix = np.sqrt(np.diag(np.matrix(covMatrix))) * param
    old_x = np.array(old_x)
    new_x = np.array(new_x)
    delta_x = old_x-new_x

    return 0.5 * delta_x**2 * covMatrix


def price_impact(old_x,new_x,covMatrix,param=0.05,phi=0.5):
    isinstance(param,float)
    isinstance(phi,float)

    covMatrix = np.sqrt(np.diag(np.matrix(covMatrix))) * param
    old_x = np.array(old_x)
    new_x = np.array(new_x)
    delta_x = old_x-new_x

    return phi * new_x * covMatrix * delta_x - old_x * covMatrix * delta_x - 0.5 * delta_x**2 * covMatrix

def reshape_cab(df_list):
    '''
    reshape list of df shape (A,B,C) to (C,A,B)
    for example, (144,22,13) to (13,144,22)
    :param df_list:
    :return:
    '''
    assert isinstance(df_list,list)
    assert isinstance(df_list[0],pd.DataFrame)
    reshape =[]
    for i in range(len(df_list[0].columns)):
        strat = []
        for df in df_list:
            strat.append(df.iloc[:,i])
        strat = pd.DataFrame(strat)
        reshape.append(strat)
    return reshape

def ex_post_return(ex_ante,window,strat_weight,factor_etf):
    assert isinstance(ex_ante,pd.DataFrame)
    assert isinstance(factor_etf,pd.DataFrame)
    return_expost=[]
    for idx in range(len(ex_ante.columns)):
        strat_penalty=[]
        strat_ex_post=[]
        strat_ex_post.append(ex_ante.iloc[:,idx][0])#we assume at period 0, no transaction_cost or price_impact
        for i in range(1,len(factor_etf)-window):
            cov_matrix=factor_etf.iloc[i:i+window].cov()
            new_x=strat_weight[idx].iloc[i]
            old_x=strat_weight[idx].iloc[i-1]
            tc = transaction_cost(old_x,new_x,cov_matrix)
            pi=price_impact(old_x,new_x,cov_matrix)
            penalty = tc+pi
            strat_penalty.append(penalty.sum())
        for i in range(1,len(ex_ante)):
            strat_ex_post.append(ex_ante.iloc[:,idx][i]+strat_penalty[i-1])
        return_expost.append(strat_ex_post)
    return pd.DataFrame(return_expost,columns=ex_ante.index,index=ex_ante.columns).T

def factor_hf_split(arr,split_pos,reshape=True):
    '''
    factor needs to be infront of hf return
    split_pos will be included in factor
    :param arr: np array
    :param split_pos: int
    :return:
    '''
    assert isinstance(arr,np.ndarray)
    assert isinstance(split_pos,int)
    assert arr.ndim==3
    assert 0<split_pos<arr.shape[2]
    factor,hf=[],[]
    for i in range(arr.shape[0]):
        factor.append(arr[i][:,:split_pos])
        hf.append(arr[i][:,split_pos:])
    factor,hf= np.array(factor),np.array(hf)
    if reshape:
        factor = factor.reshape(factor.shape[0]*factor.shape[1],factor.shape[2])
        hf = hf.reshape(hf.shape[0]*hf.shape[1],hf.shape[2])
    return factor,hf

def dic_save(dic,loc):
    a_file = open(loc,'wb')
    pickle.dump(dic,a_file)
    a_file.close()
    # test if readable
    output = dic_read(loc)
    print('stored dictionary:\n')
    print(output)