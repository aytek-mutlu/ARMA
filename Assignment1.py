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
    
    ## clean data
    
    #take natural log
    df = np.log(df)
    ##TO BE COMPLETED
    
    
    
    
    return df

def plot_summarize_data(df):
    '''
    Purpose:
        Plot and summarize data
    
    Inputs:
        df:     dataframe, data
        
    '''
    ##plot volumes
    fig,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(30,3))
    ax1.plot(df.iloc[:,0])
    ax1.set_title('MSFT Volume')
    ax2.plot(df.iloc[:,1])
    ax2.set_title('IBM Volume')
    ax3.plot(df.iloc[:,2])
    ax3.set_title('AAPL Volume')
    plt.savefig('plots.png')
    
    ##TO BE DONE -> correct digits
    
    
    ##summarize data
    df_desc = describe(df)
    df_summary = pd.DataFrame(columns = df.columns,index=['Min','Max','Mean','Std. Deviation','Skewness','Kurtosis'])
    df_summary.loc['Min'] = df_desc[1][0]
    df_summary.loc['Max'] = df_desc[1][1]
    df_summary.loc['Mean'] = df_desc[2]
    df_summary.loc['Std. Deviation'] = np.sqrt(df_desc[3])
    df_summary.loc['Skewness'] = df_desc[4]
    df_summary.loc['Kurtosis'] = df_desc[5]
    
def LL_PredictionErrorDecomposition(vP,vY,p,q):
    
    #parameter decomposition
    mu = vP[0]
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
            vE[i] = vY[i] - mu  - vEp - vEq
        
    ##change this, not start from 0 but from max(p,q) the rest will be calculated different
    return -0.5*(np.log(2*math.pi*(sigma**2))*iY + np.sum(vE*vE/(sigma**2)))
    ##for initial values, assume stationarity and therefore use unconditional mean
    ## place formula and then calculate unconditional mean for initial values
    ## initial m is (1-sum(phis))/mu
    ## initial variance is difference
        

    
def ARMA_compute(df,stock,year,p,q):
    vY = df.loc[str(year):str(year+10),stock]
    
    #list OF initial parameters based on p and q
    vP0 = np.ones(p+q+2)
    
    #minimizing function
    avgLL= lambda vP: -LL_PredictionErrorDecomposition(vP, vY,p,q)  
    res= opt.minimize(avgLL, vP0, method="BFGS")
    print('Parameters are estimated by ML for '+stock+' for decade '+ str(year) + ' with p: '+str(p)+' and q: ' + str(q)+'\n')
    #parameter estimates
    phis = res.x[1:p+1]
    mu = res.x[0]
    thetas = res.x[p+1:p+q+1]
    params = [mu,phis,thetas]
    
    #scores
    aic = res.fun*2+2*len(res.x)
    bic = np.log(len(vY))*len(res.x) + 2*res.fun
    scores = [aic,bic]
    
    #hessian and std. errors
    hes = -hessian_2sided(avgLL,res.x)
    cov_matrix = -np.linalg.inv(hes)
    std_errors = list(np.sqrt(np.diag(cov_matrix)))
    
    return [scores,params,std_errors]


def ARMA_package(df,stock,year,p,q):
    
    vY = df.loc[str(year):str(year+10),stock]
    model  = ARMA(vY.values,(p,q)).fit(disp=False,trend='c')
    print('Parameters are estimated with package for '+stock+' for decade '+ str(year) + ' with p: '+str(p)+' and q: ' + str(q)+'\n')

    #parameter estimates
    phis = model.params[1:p+1]
    
    #!!!!!!!!!!!!!!!!!!!explain this better
    mu = (1-np.sum(phis))*model.params[0]
    #!!!!!!!!!
    thetas = model.params[p+1:p+q+1]
    params = [mu,phis,thetas]

    #scores
    scores = [model.aic,model.bic]
    
    #std. errors
    std_errors = list(np.sqrt(np.diag(model.cov_params())))
    
    return [scores,params,std_errors]

    
### main
def main():
    
    stocks = ['MSFT','IBM','AAPL']
    filename = 'volumes'
    ##read and clean data
    df = read_clean_data(stocks,filename)
    
    ##plot and extract main descriptives
    plot_summarize_data(df)
    
    #data and model input combinations
    stocks = list(df.columns)
    years = [1990,2000,2010]
    ps=[0,1]
    qs=[0,1]
    
    #output dfs
    results  = pd.DataFrame(index = pd.MultiIndex.from_product([stocks,years,ps,qs]),columns=['scores','params','std_errors'])
    results_package  = copy.deepcopy(results)
    

    #ARMA estimates by computation and package
    for stock,year,p,q in results.index :
        ##ARMA by computation
        results.loc[stock,year,p,q] = ARMA_compute(df,stock,year,p,q)
        ##ARMA by packages
        results_package.loc[stock,year,p,q] = ARMA_package(df,stock,year,p,q)



### start main
if __name__ == "__main__":
    main()