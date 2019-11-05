# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 09:47:54 2019

@author: aytekm
"""

import pandas as pd
import numpy as np
from scipy.stats import describe
import matplotlib.pyplot as plt
import scipy.optimize as opt
from statsmodels.tsa.arima_model import ARMA
import statsmodels.api as sm
import math
import copy

from lib.grad import *

def read_clean_data(stocks,filename):
    '''
    Purpose:
        Read and clean volume data of given stocks
    
    Inputs:
        stocks:     list, with list of stocks
        filename:   string, filename for volumes
        
    Return value:
        df          dataframe, data
    '''
    ## read data and extract volume columns
    df  = pd.read_csv('data/'+filename+'.csv', index_col='date',parse_dates=['date'],dayfirst=False)
    df = df[[s+'-V' for s in stocks]]
    
    ##take natural log
    df = np.log(df)
    
    ##plot data initially
    df.plot(figsize=(10,3))
    
    ## DATA CLEANING
    
    #no missing values
    
    print(df.info())
    
    #histogram of data
    df.plot.hist(bins=30)
    
    #check outliers
    inter_quartile_range = (df.describe().loc['75%'] - df.describe().loc['25%']).values
    outlier_min_limit = df.describe().loc['25%'].values - 1.5*inter_quartile_range
    outlier_max_limit = df.describe().loc['75%'].values + 1.5*inter_quartile_range
    outliers = df[np.logical_or((df>outlier_max_limit).values,(df<outlier_min_limit).values)]
    
    #outliers is empty so there are no outliers according to Tukey test

    return df

def plot_summarize_data(df):
    '''
    Purpose:
        Plot and summarize data
    
    Inputs:
        df:     dataframe, data
        
    '''
    ##plot volumes
    fig,ax = plt.subplots(figsize=(10,3))
    ax.plot(df)
    ax.set_title(df.columns[0]+' Volume')
    plt.savefig('plots.png')
    
    ##TO BE DONE -> correct digits
    
    
    ##summarize data
    df_desc = describe(df)
    df_summary = pd.DataFrame(columns = df.columns,index=['Mean','Std. Deviation','Min','Max','Skewness','Kurtosis'])

    df_summary.loc['Mean'] = df_desc[2]
    df_summary.loc['Std. Deviation'] = np.sqrt(df_desc[3])
    df_summary.loc['Min'] = df_desc[1][0]
    df_summary.loc['Max'] = df_desc[1][1]
    df_summary.loc['Skewness'] = df_desc[4]
    df_summary.loc['Kurtosis'] = df_desc[5]
    
    return df_summary
    
def LL_PredictionErrorDecomposition(vP,vY,p,q):
    
    #parameter decomposition
    m = vP[0]
    phis = vP[1:p+1]
    thetas = vP[p+1:p+q+1]
    sigma = vP[-1]
    
    p = len(phis)
    q = len(thetas)
    
    #number of observations
    iY = len(vY)
    
    #initiation of residuls matrix
    vE = np.zeros(iY)
    
    for i in range(iY):
        #contribution of AR terms to vE
        vEp = 0
        if i>=p:
            for j in range(1,p+1):
                vEp += vY[i-j]*phis[j-1]
        
        #contribution of MA terms to vE
        vEq = 0
        if i>=q:
            for j in range(1,q+1):
                vEq += vE[i-j]*thetas[j-1]
      
        #combine both effects to find vE
        if i>0:
            vE[i] = vY[i] - m  - vEp - vEq
        
    return -0.5*(np.log(2*math.pi*(sigma**2))*iY + np.sum(vE*vE/(sigma**2)))

    
def ARMA_compute(df,stock,year,p,q,vP):
    vY = df.loc[str(year):str(year+10),stock]
    
    #list of initial parameters based on p and q
    if (p==1) & (q==1) :
        vP0 = [vP[0],1,vP[2][0],vP[3]]
    elif (p==0) & (q==0):
        vP0 = [vP[0],vP[3]]
    elif (p==0) & (q==1):
        vP0 = [vP[0],1,vP[3]]
    elif (p==1) & (q==0):
        vP0 = [vP[0],1,vP[3]]

    
    #minimizing function
    sumLL= lambda vP: -LL_PredictionErrorDecomposition(vP,vY,p,q)  
    res= opt.minimize(sumLL, vP0, method="BFGS")
    print('Parameters are estimated by ML for '+stock+' for decade '+ str(year) + ' with p: '+str(p)+' and q: ' + str(q)+'\n')
    #parameter estimates
    phis = res.x[1:p+1]
    m = res.x[0]
    thetas = res.x[p+1:p+q+1]
    sigma = res.x[-1]
    params = [m,phis,thetas,sigma]
    
    #scores
    aic = res.fun*2+2*len(res.x)
    bic = np.log(len(vY))*len(res.x) + 2*res.fun
    scores = [aic,bic]
    
    #hessian and std. errors
    hes = -hessian_2sided(sumLL,res.x)
    cov_matrix = -np.linalg.inv(hes)
    std_errors = list(np.sqrt(np.diag(cov_matrix)))
    
    #sandwich form robust std. errors
    hes = -hessian_2sided(sumLL,res.x)
    mHI = np.linalg.inv(hes)
    mHI = (mHI + mHI.T)/2
    mG = jacobian_2sided(sumLL,res.x)
    
    cov_matrix_sandwich = mHI @ (mG.T @ mG) @ mHI
    std_errors_sandwich = list(np.sqrt(np.diag(cov_matrix_sandwich)))

    
    return [scores,params,std_errors],params


def ARMA_package(df,stock,year,p,q):
    
    vY = df.loc[str(year):str(year+10),stock]
    model  = ARMA(vY.values,(p,q)).fit(disp=False,trend='c')
    print('Parameters are estimated with package for '+stock+' for decade '+ str(year) + ' with p: '+str(p)+' and q: ' + str(q)+'\n')

    #parameter estimates
    phis = model.params[1:p+1]
    
    m = (1-np.sum(phis))*model.params[0]
    thetas = model.params[p+1:p+q+1]
    sigma = np.sqrt(model.sigma2)
    params = [m,phis,thetas,sigma]

    #scores
    scores = [model.aic,model.bic]
    
    #std. errors
    std_errors = list(np.sqrt(np.diag(model.cov_params())))
    
    return [scores,params,std_errors]


def interpret_results(df):
    
    df.reset_index(inplace=True)
    df.rename(columns={'level_0':'Stock','level_1':'Decade','level_2':'p','level_3':'q'},inplace=True)
    
    df[['AIC','BIC']] = pd.DataFrame(df.scores.values.tolist(), index= df.index)
    df['m'] = [i[0] for i in df.params]
    df['m_se'] = [i[0] for i in df.std_errors]
    
    
    for j in range(max(df.p)):
        df['phi_'+str(j+1)]= [param[1][j] if p>=(j+1) else np.nan for param,p in zip(df.params,df.p) ]
        df['phi_'+str(j+1)+'_se']= [std_errors[j+1] if p>=(j+1) else np.nan for std_errors,p in zip(df.std_errors,df.p) ]
        
    for j in range(max(df.q)):
        df['theta_'+str(j+1)]= [param[2][j] if q>=(j+1) else np.nan for param,q in zip(df.params,df.q)]
        df['theta_'+str(j+1)+'_se']= [std_errors[j+p+1] if q>=(j+1) else np.nan for std_errors,p,q in zip(df.std_errors,df.p,df.q) ]

    df['sigma_squared'] = [pow(i[3],2) for i in df.params]
    df['sigma_squared_se'] = [i[-1] for i in df.std_errors]

    
    df.drop(columns=['scores','params','std_errors'],inplace=True)
        
    return df
    
    
    
### main
def main():
    
    stocks = ['AAPL']
    filename = 'volumes'
    ##read and clean data
    df = read_clean_data(stocks,filename)
    
    ##plot and extract main descriptives
    print(plot_summarize_data(df))
    
    #data and model input combinations
    stocks = list(df.columns)
    years = [1990,2000,2010]
    
    #list of possible p and q values    
    p_list=[0,1]
    q_list=[0,1]
    
    #output dfs
    results  = pd.DataFrame(index = pd.MultiIndex.from_product([stocks,years,p_list,q_list]),columns=['scores','params','std_errors'])
    results_package  = copy.deepcopy(results)
    stock = stocks[0]    

    #ARMA estimates by computation and package
    for year in years:
        vP = [np.ones(1),[],[],np.ones(1)]
        ##ARMA by computation
        results.loc[stock,year,0,0],vP_noise = ARMA_compute(df,stock,year,0,0,vP)
        results.loc[stock,year,0,1],vP_01 = ARMA_compute(df,stock,year,0,1,vP_noise)
        results.loc[stock,year,1,0],vP_10 = ARMA_compute(df,stock,year,1,0,vP_noise)
        results.loc[stock,year,1,1],vP_11 = ARMA_compute(df,stock,year,1,1,vP_01)

        ##ARMA by packages
        results_package.loc[stock,year,0,0] = ARMA_package(df,stock,year,0,0)
        results_package.loc[stock,year,0,1] = ARMA_package(df,stock,year,0,1)
        results_package.loc[stock,year,1,0] = ARMA_package(df,stock,year,1,0)
        results_package.loc[stock,year,1,1] = ARMA_package(df,stock,year,1,1)


    results_final = interpret_results(results)
    results_package_final = interpret_results(results_package)

    #TODO:
    #calculating std. errors with robust sandwich

### start main
if __name__ == "__main__":
    main()