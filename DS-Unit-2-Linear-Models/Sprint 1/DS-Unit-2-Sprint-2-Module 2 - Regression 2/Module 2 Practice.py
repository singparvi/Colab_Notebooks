# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np
import seaborn as sns

#load the data into a DataFrame
penguins = sns.load_dataset('penguins')

# Drop Nan
penguins.dropna(inplace = True)

# Create the 2-D features matrix
X_penguins = penguins['flipper_length_mm']
X_penguins_2D = X_penguins[:,np.newaxis]

# Create the target array
y_penguins = penguins['body_mass_g']

# %%
# import the train_test_split utility
from sklearn.model_selection import train_test_split

# creating the training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_penguins_2D, y_penguins, test_size = 0.2, random_state = 42)

print ('The training and testing feature:', X_train.shape, X_test.shape)
print ('The training and testing target:', y_train.shape, y_test.shape)


# %%
# import the predictor class

from sklearn.linear_model import LinearRegression

# instantaniate the class ( with default parameters)
model = LinearRegression()

# fit the model
model.fit(X_train, y_train)

# %%

# slope (also called as model coefficient)
print(model.coef_)

# intercept 
print (model.intercept_)

#print the equation
print (f'body_mass_g = {model.intercept_} + {model.coef_[0]}xflipper_length_mm')


# %%
#making the prediction
# we'll use a test set and the model prediction and look at the r2_score or R-squared value. R-squared is a statistical measure of how close the data are to the fitted regression line. A value of 100% (or 1) means that all of the variation around the mean is explained by the model; the best-fit line would go through all of the data points.

#use the test set for predictions
y_predict = model.predict(X_test)

#calculate the accuracy score
from sklearn.metrics import r2_score
r2_score(y_test, y_predict)
# %% [markdown]

Objective 02 - Use scikit-learn to fit a multiple regression


# %%
#import pandas and seaborn

import pandas as pd
import seaborn as sns
import numpy as np

# load the data into a DF
penguins = sns.load_dataset('penguins')

display(penguins.head())

# %%

#import 
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1,2,figsize = (8,4))
sns.scatterplot(x = 'flipper_length_mm', y = 'body_mass_g', data = penguins, ax = ax1)
sns.scatterplot(x = 'bill_length_mm', y = 'body_mass_g', data = penguins, ax = ax2)

plt.show()

# %%

#remove the nan value
penguins.dropna(inplace=True)

# create the 2-D feature matrix
features =  ['bill_length_mm', 'flipper_length_mm']
X_penguins = penguins[features]

# create the target array
y_penguins = penguins['body_mass_g']

# import the estimator class
from sklearn.linear_model import LinearRegression

# instantiate the class
model = LinearRegression()

# fit the model 
model.fit(X_penguins, y_penguins)
# %%

#slope and coefficient be
print(f'Outputs are {model.intercept_} and  {model.coef_[0]}')


# %%
# NOT LEARNT BUT PASTED AS IS

# Format the data for plotting
x_flipper = penguins['flipper_length_mm']
y_culmen = penguins['bill_length_mm']
z_weight = penguins['body_mass_g']

# Create the data to plot the best-fit plane
(x_plane, y_plane) = np.meshgrid(np.arange(165, 235, 1), np.arange(30, 60, 1))
z_plane = -5836 + 49*x_plane + 5*y_plane

# Import for plotting
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Initial the figure and axes objects
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')

# Plot the data: 2 features, 1 target
ax.scatter(xs=x_flipper, ys=y_culmen, zs=z_weight, zdir='z', 
           s=20, c=z_weight, cmap=cm.viridis)

# Plot the best-fit plane
ax.plot_surface(x_plane, y_plane, z_plane, color='gray', alpha = 0.5)

# General figure/axes properties
ax.view_init(elev=28, azim=325)
ax.set_xlabel('Flipper length')
ax.set_ylabel('Culmen length')
ax.set_zlabel('Body mass')
fig.tight_layout()

plt.show()
# plt.clf()

# %% [markdown]

Objective 03 - Understand how ordinary least squares regression minimizes the sum of squared errors

# %%


# %%

# %%

# %%
