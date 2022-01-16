#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import statsmodels.api as sm
import numpy as np


# # Data Import and Preparation

# In[2]:


data_df = pd.read_excel('Stepwise_JJ.xlsx', index_col=0,sheet_name="Python Data", engine="openpyxl")
data_df.head(1)


# # Data Overview

# In[11]:


def regression_plots(regression_data_df = data_df, y_variable = "RoE Japan", columns = 3,savefig=True):
    rows = int(np.ceil((len(regression_data_df.columns)-1)/columns))
    fig,axs = plt.subplots(rows,columns, sharey=True,figsize = (rows*4,rows*4))
    for col,ax in enumerate(axs.flatten()):
        if col <len(regression_data_df.columns):
            column = regression_data_df.columns[col]
            if column == y_variable:
                continue
            else:
                sns.regplot(x = regression_data_df[column],y=regression_data_df[y_variable],ax=ax,robust=False)
    fig.tight_layout()
    if savefig==True:
        fig.savefig("Filtered Variables.png")    


# In[9]:


columns = 4
rows = int(np.ceil((len(data_df.columns)-1)/columns))
fig,axs = plt.subplots(rows,columns, sharey=True,figsize = (16,16))
for col,ax in enumerate(axs.flatten()):
    if col+1 <len(data_df.columns):
        column = data_df.columns[col+1]
        #print(data_df[column].head(2))
        sns.regplot(x = data_df[column],y=data_df["RoE Japan"],ax=ax,robust=False)
fig.tight_layout()
fig.savefig("Data Overview.png")


# # Stepwise Regression

# In[5]:


def stepwise_regression(regression_data_df = data_df, y_variable = "RoE Japan",constant = True, max_p = 0.05,only_positive=True):
    '''
    This function iterates over the given data frame, trying to explain the y variable by all variables except iteself.
    The process stops once all variables are below the specified p-value. 
    By default only positive coefficients are accepted. Model filters until no negative coefficients are in the regression anymore.
    '''
    
    #Set Original Dataframe
    X = regression_data_df
    X = X.drop([y_variable],axis=1)
    if constant == True:
        X = sm.add_constant(X)
    Y = regression_data_df[y_variable]
    exog_vars = len(X.columns)
    
    #Iterate
    while exog_vars >2:
        model = sm.OLS(endog = Y, exog = X)
        results = model.fit()
        
        worst_p_value = results.pvalues.max()
        worst_p_name = results.pvalues.idxmax()
        X = X.drop([worst_p_name],axis=1)
        exog_vars = len(X.columns)
        if worst_p_value<max_p:
            break

    model = sm.OLS(endog = Y, exog = X)
    results = model.fit()
    params_inc_neg = results.params
    
    if only_positive==True:         
        while len(results.params[results.params<0])>0:
            coeffs = results.params
            coeffs_pos = coeffs[coeffs>0]
            X = X[coeffs_pos.index.to_list()]
            model = sm.OLS(endog = Y, exog = X)
            results = model.fit()

    relevant_vars = regression_data_df[X.columns.drop("const").to_list()+[y_variable]]
    return results, relevant_vars,params_inc_neg


# In[21]:


stepwise_regression(regression_data_df = data_df, y_variable = "RoE Japan",constant = True, max_p = 0.05,only_positive=True)[0].summary()
file = open("regression_output.text","w")
file.write(stepwise_regression(regression_data_df = data_df, y_variable = "RoE Japan",constant = True, max_p = 0.05,only_positive=True)[0].summary().as_text())
file.close()


# In[7]:


filtered_var_df = stepwise_regression(regression_data_df = data_df, y_variable = "RoE Japan",constant = True, max_p = 0.05)[1]
filtered_var_df.head(2)


# In[13]:


regression_plots(regression_data_df = filtered_var_df, y_variable = "RoE Japan", columns = 1,savefig=True)

