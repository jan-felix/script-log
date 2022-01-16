#!/usr/bin/env python
# coding: utf-8

# In[27]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import statsmodels.api as sm
import numpy as np
import streamlit as st 


# # Data Import and Preparation

# In[31]:


st.title("Stepwise Regression Calculator")

uploaded_file = st.file_uploader("Chose a CSV file that contains the date for the regression.")
st.text("This must be a csv file that his build like below:")

sample_df = pd.DataFrame({"Variable A":[1,2,3],"Variable B":[2,6,1],"Further Variables":[5,6,7]},index=pd.DatetimeIndex(["31/12/2020","31/12/2021","31/12/2022"]))
sample_df.head(3)
st.text("The first column of the csv must have dates, the first rows must have variables. Each variable has its own column.")


# In[43]:


if uploaded_file is not None:
    data_df = pd.read_csv(uploaded_file,index_col=0)
    data_df.index = pd.DatetimeIndex(data_df.index)
    st.write("Does this look right to you?")
    st.write(dataframe.head())
y_var = st.selectbox(label="What is your endogenous variable?",options=data_df.columns.to_list())


# # Data Overview

# In[37]:


def regression_plots(regression_data_df = data_df, y_variable = y_var, columns = 3):
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
    return fig 


# In[47]:


st.text("Lets take a first look at the individual relationships.")

columns = st.selectbox("How many columns of charts should the below have?", options=[1,2,3,4,5],index=3)


rows = int(np.ceil((len(data_df.columns)-1)/columns))
fig,axs = plt.subplots(rows,columns, sharey=True,figsize = (16,16))
for col,ax in enumerate(axs.flatten()):
    if col+1 <len(data_df.columns):
        column = data_df.columns[col+1]
        sns.regplot(x = data_df[column],y=data_df[y_var],ax=ax,robust=False)
fig.tight_layout()
st.pyplot(fig=fig)


# # Stepwise Regression

# In[5]:


def stepwise_regression(regression_data_df = data_df, y_variable = y_var,constant = True, max_p = 0.05,only_positive=True):
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


st.text("Below you see the regression results")
stepwise_regression(regression_data_df = data_df, y_variable = y_var,constant = True, max_p = 0.05,only_positive=True)[0].summary()


# In[48]:



columns = st.selectbox("How many columns of charts should the below have?", options=[1,2,3,4,5],index=3)

reg_fig = regression_plots(regression_data_df = filtered_var_df, y_variable = "RoE Japan", columns = 1,savefig=True)
st.pyplot(fig=reg_fig)

