# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# Lambda School Data Science
# 
# *Unit 2, Sprint 1, Module 1*
# 
# ---
# %% [markdown]
# # Regression 1
# 
# ## Assignment
# 
# You'll use another **New York City** real estate dataset. 
# 
# But now you'll **predict how much it costs to rent an apartment**, instead of how much it costs to buy a condo.
# 
# The data comes from renthop.com, an apartment listing website.
# 
# - [ ] Look at the data. Choose a feature, and plot its relationship with the target.
# - [ ] Use scikit-learn for linear regression with one feature. You can follow the [5-step process from Jake VanderPlas](https://jakevdp.github.io/PythonDataScienceHandbook/05.02-introducing-scikit-learn.html#Basics-of-the-API).
# - [ ] Define a function to make new predictions and explain the model coefficient.
# - [ ] Organize and comment your code.
# 
# > [Do Not Copy-Paste.](https://docs.google.com/document/d/1ubOw9B3Hfip27hF2ZFnW3a3z9xAgrUDRReOEo-FHCVs/edit) You must type each of these exercises in, manually. If you copy and paste, you might as well not even do them. The point of these exercises is to train your hands, your brain, and your mind in how to read, write, and see code. If you copy-paste, you are cheating yourself out of the effectiveness of the lessons.
# 
# If your **Plotly** visualizations aren't working:
# - You must have JavaScript enabled in your browser
# - You probably want to use Chrome or Firefox
# - You may need to turn off ad blockers
# - [If you're using Jupyter Lab locally, you need to install some "extensions"](https://plot.ly/python/getting-started/#jupyterlab-support-python-35)
# 
# ## Stretch Goals
# - [ ] Do linear regression with two or more features.
# - [ ] Read [The Discovery of Statistical Regression](https://priceonomics.com/the-discovery-of-statistical-regression/)
# - [ ] Read [_An Introduction to Statistical Learning_](http://faculty.marshall.usc.edu/gareth-james/ISL/ISLR%20Seventh%20Printing.pdf), Chapter 2.1: What Is Statistical Learning?

# %%
import sys

# If you're on Colab:
if 'google.colab' in sys.modules:
    DATA_PATH = 'https://raw.githubusercontent.com/LambdaSchool/DS-Unit-2-Applied-Modeling/master/data/'

# If you're working locally:
else:
    DATA_PATH = '../data/'
    
# Ignore this Numpy warning when using Plotly Express:
# FutureWarning: Method .ptp is deprecated and will be removed in a future version. Use numpy.ptp instead.
import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning, module='numpy')


# %%
# Read New York City apartment rental listing data
import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/LambdaSchool/DS-Unit-2-Applied-Modeling/master/data/condos/tribeca.csv',
parse_dates = ['SALE_DATE'], 
index_col = 'SALE_DATE',
sum_r {'ZIP_CODE': 'object','SALE_PRICE':'float','YEAR_BUILT':'int'})


# %%
df.shape

# %% [markdown]
# ## Code by Parvi from here on --->

# %%
df.info()


# %%
df.head()


# %%
import matplotlib.pyplot as plt

df['GROSS_SQUARE_FEET'].hist()
plt.title('Distribution for Condo Size')
plt.xlabel('Gross Square Feet')
plt.ylabel('Frequency');


# %%
import seaborn as sns

df['SALE_PRICE'].hist()
plt.xlabel('Price [$ 10 Million]')
plt.ylabel('Count')
plt.title('Distribution of Condo Prices');


# %%


# %% [markdown]
# 

# %%



# %%



# %%



# %%



# %%



# %%



# %%



# %%



# %%



# %%
# finding the mean_absolute_error
mean_guess = df.price.mean()

# print (mean_guess)

#does not make a difference whether you choose mean_guess before or after in 
# calculation
error = mean_guess - df.price
mean_absolute_error = error.abs().mean()

print (f'The mean absolute error when using mean as the baseline is {mean_absolute_error}')

# %% [markdown]
# ## The dataframe seem to be complete with no 'key' null value rows. Though there are some null values in description, display_address and street_address but they will not be a part of our model, therefore, should not be an issue.

# %%
#printing the column names

df.columns


# %%
# first plotting the datasets of our interst as scatter plot and 
# regression fitting with X as bedrooms and y as price

import plotly.express as px

px.scatter(data_frame= df, x = 'bedrooms', y = 'price', trendline= 'ols')


# %%
#features extraction from the columns we know that we need to 
# have a model where X is and y is 'price'
import numpy as np
X_train_1D = df['bedrooms']
#note X_train must be a 2D array therefore
X_train = X_train_1D[:,np.newaxis]

# target vector being price
y_train = df['price']

# shapes
print (f'X shape is {X_train.shape} while y shape is {y_train.shape}')


# %%
# choosing the class of model

from sklearn.linear_model import LinearRegression

# instantiate (make a) class model

model = LinearRegression()
model


# %%
# fitting the model in X,y format
model.fit(X_train, y_train)


# %%
# print out the parameters of the fit model
print (f'y = {model.intercept_} + {model.coef_[0]}X')


# %%
# predicting using the model 
y_predict_bedroom = [[8]]
y_predict = model.predict(y_predict_bedroom)

print (f'For a {y_predict_bedroom[0][0]} bedroom the model predicts a rent of ${y_predict[0]}')


# %%
# finding the price of a 8 bedroom apt to compare prediction with
condition = (df.bedrooms == 8)

df[condition]


# %%
# finding the MAE from sklearn

from sklearn.metrics import mean_absolute_error
y_test = [9995]
mean_absolute_error(y_predict, y_test)

# %% [markdown]
# ---
# %% [markdown]
# # The following section is to do a multiple regression i.e. including more than one variable in feature matrix

# %%
df.columns


# %%
# feature engineering picking 
X = df[['bathrooms', 'bedrooms']]

# the value of y has not changed therefore no need to make target vector again
# in case needed do 
y = df['price'] 
# to get the target vector


# %%
# fitting the model with the instantiated class
# no need to instantiate class again as already done above
model.fit(X,y)


# %%
# predict using the model
y_pred_X = [[3,8]]
y_pred = model.predict(y_pred_X)

#printing the results
print (f'For a NYC apartment with {y_pred_X[0][0]} bathrooms and {y_pred_X[0][1]} bedrooms the predicted price is ${y_pred[0]}')


# %%
# calculating the MAE
from sklearn.metrics import mean_absolute_error
y_test = [9995]
mean_absolute_error(y_pred, y_test)

# %% [markdown]
# ## Pretty close to the actual price huh

