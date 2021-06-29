# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# Objective 01 - Begin with baselines for regression
# 

# %%
# import seaborn and matplotlib libraries 

import seaborn as sns
import matplotlib.pyplot as plt

# load the example in the penguin dataset
penguins = sns.load_dataset('penguins')

# create a regplot
sns.regplot(x = 'flipper_length_mm', y= 'body_mass_g', data= penguins,fit_reg=True)

plt.show()




# %%
# plot the same data as above with  added lines for our"guess"

ax = sns.regplot(x = 'flipper_length_mm', y = 'body_mass_g', data=penguins, fit_reg=True)
plt.axvline(x = 190, color = 'red', linewidth = 0.75)
plt.axhline(y = 4850, color = 'red', linewidth = 0.75)

plt.show()
# %% [markdown]
# Objective 02 - Use scikit-learn for linear regression
# 
# %%
# import pandas and seaborn

import pandas as pd
import seaborn as sns

import numpy as np

# load the data into a DF

penguins = sns.load_dataset('penguins')

# print the shape of the DF
print (penguins.shape)

# drop NaN values

penguins.dropna(inplace=True)

# print the shape of the DF
print (penguins.shape)

penguins.head()



# %%

# create the feature matrix
X_penguins = penguins['flipper_length_mm']
print ('The shape of feature matrix:', X_penguins.shape)

y_penguins = penguins['body_mass_g']
print ('The shape of target vector matrix:', y_penguins.shape)

# %%
# import predictor class
from sklearn.linear_model import LinearRegression

# instantiate the class with default parameters
model = LinearRegression()

# display the model parameters
print (model)

# %%
# Display the shape of X_penguins
print ('Original features matrix:', X_penguins.shape)

# add a new axis to create a column venctor
X_penguins_2D = X_penguins[:,np.newaxis]

print (X_penguins_2D.shape)
# %%
#fit the model

model.fit(X_penguins_2D, y_penguins)
# %%
# slope (also called as model coefficient)

print (model.coef_)

# intercept

print (model.intercept_)

# in equation format

print (f'\n body_mass_g = {model.coef_[0]} x flipper_length_mm + ({model.intercept_})')

# %% [markdown]
# Objective 03 - Explain the coefficients from a linear regression

# 

# %%
x_line = np.linspace(170, 240)
y_line = model.coef_*x_line + model.intercept_


# %%
# improt plotting libraries

import matplotlib.pyplot as plt

# create fig and ax objects
fig, ax = plt.subplots(1)

ax.scatter(x = X_penguins, y = y_penguins, label = 'Observed Data')
ax.plot(x_line, y_line, color = 'g', label = 'linear regression model')
ax.set_xlabel('Penguin flipper length (mm)')
ax.set_ylabel('Penguin weight (g)')
ax.legend()

plt.show()

# %%

import pandas as pd