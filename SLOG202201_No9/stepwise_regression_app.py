#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
import streamlit as st 


# # Helper Functions

# In[12]:


data_df = pd.DataFrame()
y_var = ""


# In[13]:


def regression_plots(regression_data_df = data_df, y_variable = y_var, columns = 3):
    
    rows = int(np.ceil((len(regression_data_df.columns)-1)/columns))
    fig,axs = plt.subplots(rows,columns, sharey=True,figsize = (rows*4,rows*4))
    if len(axs)==1:
        sns.regplot(x = regression_data_df[regression_data_df.columns.drop(y_variable).to_list()],
                    y=regression_data_df[y_variable],ax=axs,robust=False)
    
    for col,ax in enumerate(axs.flatten()):
        if col <len(regression_data_df.columns):
            column = regression_data_df.columns[col]
            if column == y_variable:
                continue
            else:
                sns.regplot(x = regression_data_df[column],y=regression_data_df[y_variable],ax=ax,robust=False)
    fig.tight_layout()
    return fig 

def stepwise_regression(regression_data_df = data_df, y_variable = y_var,constant = True, max_p = 0.05,only_positive=True):
    '''
    This function iterates over the given data frame, trying to explain the y variable by all variables except itself.
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
            
    if "const" in X.columns:
        relevant_vars = regression_data_df[X.columns.drop("const").to_list()+[y_variable]]
    else:
        relevant_vars = regression_data_df[X.columns.to_list()+[y_variable]]
    
    return results, relevant_vars,params_inc_neg


# # Intro

# In[14]:


st.title("Stepwise Regression Calculator")


# In[15]:


uploaded_file = st.file_uploader("Chose a CSV file that contains the date for the regression.")
st.text("This must be a csv file that his build like below:")


# In[16]:


sample_df = pd.DataFrame({"Variable A":[1,2,3],"Variable B":[2,6,1],"Further Variables":[5,6,7]},index=pd.DatetimeIndex(["31/12/2020","31/12/2021","31/12/2022"]))
st.table(sample_df)
st.text("The first column of the csv must have dates, the first rows must have variables. Each variable has its own column.")


# # Process

# In[17]:


format_dec = st.selectbox("German or English Format?", options=["German","English"],key="000")


# In[18]:


if uploaded_file is not None:
    if format_dec == "German":
        data_df = pd.read_csv(uploaded_file,delimiter=";",decimal=",", index_col=0).dropna()
    if format_dec =="English":
        data_df = pd.read_csv(uploaded_file,index_col=0).dropna()
    
    data_df.index = pd.DatetimeIndex(data_df.index)
    st.write("Does this look right to you?")
    st.write(data_df.head())
    y_var = st.selectbox(label="What is your endogenous variable?",options=data_df.columns.to_list(),key="001")
    st.text("Lets take a first look at the individual relationships.")

    columns = st.selectbox("How many columns of charts should the below have?", options=[1,2,3,4,5],index=3,key="002")


    rows = int(np.ceil((len(data_df.columns)-1)/columns))
    fig,axs = plt.subplots(rows,columns, sharey=True,figsize = (16,16))
    for col,ax in enumerate(axs.flatten()):
        if col+1 <len(data_df.columns):
            column = data_df.columns[col+1]
            sns.regplot(x = data_df[column],y=data_df[y_var],ax=ax,robust=False)
    fig.tight_layout()
    st.pyplot(fig=fig)
    p_val = st.slider("At what P-Value do you want to make the cut for exclusion of a variable?",
                      min_value=0,
                      max_value=0.2,
                      value=0.05,
                      step=0.025,
                      key="slider1")
    only_positive = st.selectbox("Do you want to include only positive coefficients?",
                                 options=["Yes","No"],
                                 index=0,
                                 key="pos_neg_select")
    if only_positive == "Yes":
        only_positive = True
    else:
        only_positive = False
    constant = st.selectbox("Does your regression need a constant?",
                                     options=["Yes","No"],
                                     index=0,
                                     key="constant_select")
    if constant == "Yes":
        constant = True
    else:
        only_positive = False
    st.text("Below you see the regression results")
    st.text(stepwise_regression(regression_data_df = data_df,
                                y_variable = y_var,
                                constant = constant,
                                max_p = p_val,
                                only_positive=only_positive)[0].summary().as_text())
    
    
    filtered_var_df = stepwise_regression(regression_data_df = data_df, y_variable = y_var,constant = True, max_p = 0.05,only_positive=True)[1]
    columns = st.selectbox("How many columns of charts should the below have?", options=[1,2,3,4,5],index=3,key="003")

    reg_fig = regression_plots(regression_data_df = filtered_var_df, y_variable = y_var, columns = columns)
    st.pyplot(fig=reg_fig)

