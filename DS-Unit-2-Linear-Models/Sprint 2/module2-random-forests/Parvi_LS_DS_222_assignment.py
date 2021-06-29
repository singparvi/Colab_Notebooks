# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# Lambda School Data Science
# 
# *Unit 2, Sprint 2, Module 2*
# 
# ---
# %% [markdown]
# # Random Forests
# 
# ## Assignment
# - [ ] Read [“Adopting a Hypothesis-Driven Workflow”](http://archive.is/Nu3EI), a blog post by a Lambda DS student about the Tanzania Waterpumps challenge.
# - [ ] Continue to participate in our Kaggle challenge.
# - [ ] Define a function to wrangle train, validate, and test sets in the same way. Clean outliers and engineer features.
# - [ ] Try Ordinal Encoding.
# - [ ] Try a Random Forest Classifier.
# - [ ] Submit your predictions to our Kaggle competition. (Go to our Kaggle InClass competition webpage. Use the blue **Submit Predictions** button to upload your CSV file. Or you can use the Kaggle API to submit your predictions.)
# - [ ] Commit your notebook to your fork of the GitHub repo.
# 
# ## Stretch Goals
# 
# ### Doing
# - [ ] Add your own stretch goal(s) !
# - [ ] Do more exploratory data analysis, data cleaning, feature engineering, and feature selection.
# - [ ] Try other [categorical encodings](https://contrib.scikit-learn.org/category_encoders/).
# - [ ] Get and plot your feature importances.
# - [ ] Make visualizations and share on Slack.
# 
# ### Reading
# 
# Top recommendations in _**bold italic:**_
# 
# #### Decision Trees
# - A Visual Introduction to Machine Learning, [Part 1: A Decision Tree](http://www.r2d3.us/visual-intro-to-machine-learning-part-1/),  and _**[Part 2: Bias and Variance](http://www.r2d3.us/visual-intro-to-machine-learning-part-2/)**_
# - [Decision Trees: Advantages & Disadvantages](https://christophm.github.io/interpretable-ml-book/tree.html#advantages-2)
# - [How a Russian mathematician constructed a decision tree — by hand — to solve a medical problem](http://fastml.com/how-a-russian-mathematician-constructed-a-decision-tree-by-hand-to-solve-a-medical-problem/)
# - [How decision trees work](https://brohrer.github.io/how_decision_trees_work.html)
# - [Let’s Write a Decision Tree Classifier from Scratch](https://www.youtube.com/watch?v=LDRbO9a6XPU)
# 
# #### Random Forests
# - [_An Introduction to Statistical Learning_](http://www-bcf.usc.edu/~gareth/ISL/), Chapter 8: Tree-Based Methods
# - [Coloring with Random Forests](http://structuringtheunstructured.blogspot.com/2017/11/coloring-with-random-forests.html)
# - _**[Random Forests for Complete Beginners: The definitive guide to Random Forests and Decision Trees](https://victorzhou.com/blog/intro-to-random-forests/)**_
# 
# #### Categorical encoding for trees
# - [Are categorical variables getting lost in your random forests?](https://roamanalytics.com/2016/10/28/are-categorical-variables-getting-lost-in-your-random-forests/)
# - [Beyond One-Hot: An Exploration of Categorical Variables](http://www.willmcginnis.com/2015/11/29/beyond-one-hot-an-exploration-of-categorical-variables/)
# - _**[Categorical Features and Encoding in Decision Trees](https://medium.com/data-design/visiting-categorical-features-and-encoding-in-decision-trees-53400fa65931)**_
# - _**[Coursera — How to Win a Data Science Competition: Learn from Top Kagglers — Concept of mean encoding](https://www.coursera.org/lecture/competitive-data-science/concept-of-mean-encoding-b5Gxv)**_
# - [Mean (likelihood) encodings: a comprehensive study](https://www.kaggle.com/vprokopev/mean-likelihood-encodings-a-comprehensive-study)
# - [The Mechanics of Machine Learning, Chapter 6: Categorically Speaking](https://mlbook.explained.ai/catvars.html)
# 
# #### Imposter Syndrome
# - [Effort Shock and Reward Shock (How The Karate Kid Ruined The Modern World)](http://www.tempobook.com/2014/07/09/effort-shock-and-reward-shock/)
# - [How to manage impostor syndrome in data science](https://towardsdatascience.com/how-to-manage-impostor-syndrome-in-data-science-ad814809f068)
# - ["I am not a real data scientist"](https://brohrer.github.io/imposter_syndrome.html)
# - _**[Imposter Syndrome in Data Science](https://caitlinhudon.com/2018/01/19/imposter-syndrome-in-data-science/)**_
# 
# 
# ### More Categorical Encodings
# 
# **1.** The article **[Categorical Features and Encoding in Decision Trees](https://medium.com/data-design/visiting-categorical-features-and-encoding-in-decision-trees-53400fa65931)** mentions 4 encodings:
# 
# - **"Categorical Encoding":** This means using the raw categorical values as-is, not encoded. Scikit-learn doesn't support this, but some tree algorithm implementations do. For example, [Catboost](https://catboost.ai/), or R's [rpart](https://cran.r-project.org/web/packages/rpart/index.html) package.
# - **Numeric Encoding:** Synonymous with Label Encoding, or "Ordinal" Encoding with random order. We can use [category_encoders.OrdinalEncoder](https://contrib.scikit-learn.org/category_encoders/ordinal.html).
# - **One-Hot Encoding:** We can use [category_encoders.OneHotEncoder](https://contrib.scikit-learn.org/category_encoders/onehot.html).
# - **Binary Encoding:** We can use [category_encoders.BinaryEncoder](https://contrib.scikit-learn.org/category_encoders/binary.html).
# 
# 
# **2.** The short video 
# **[Coursera — How to Win a Data Science Competition: Learn from Top Kagglers — Concept of mean encoding](https://www.coursera.org/lecture/competitive-data-science/concept-of-mean-encoding-b5Gxv)** introduces an interesting idea: use both X _and_ y to encode categoricals.
# 
# Category Encoders has multiple implementations of this general concept:
# 
# - [CatBoost Encoder](https://contrib.scikit-learn.org/category_encoders/catboost.html)
# - [Generalized Linear Mixed Model Encoder](https://contrib.scikit-learn.org/category_encoders/glmm.html)
# - [James-Stein Encoder](https://contrib.scikit-learn.org/category_encoders/jamesstein.html)
# - [Leave One Out](https://contrib.scikit-learn.org/category_encoders/leaveoneout.html)
# - [M-estimate](https://contrib.scikit-learn.org/category_encoders/mestimate.html)
# - [Target Encoder](https://contrib.scikit-learn.org/category_encoders/targetencoder.html)
# - [Weight of Evidence](https://contrib.scikit-learn.org/category_encoders/woe.html)
# 
# Category Encoder's mean encoding implementations work for regression problems or binary classification problems. 
# 
# For multi-class classification problems, you will need to temporarily reformulate it as binary classification. For example:
# 
# ```python
# encoder = ce.TargetEncoder(min_samples_leaf=..., smoothing=...) # Both parameters > 1 to avoid overfitting
# X_train_encoded = encoder.fit_transform(X_train, y_train=='functional')
# X_val_encoded = encoder.transform(X_train, y_val=='functional')
# ```
# 
# For this reason, mean encoding won't work well within pipelines for multi-class classification problems.
# 
# **3.** The **[dirty_cat](https://dirty-cat.github.io/stable/)** library has a Target Encoder implementation that works with multi-class classification.
# 
# ```python
#  dirty_cat.TargetEncoder(clf_type='multiclass-clf')
# ```
# It also implements an interesting idea called ["Similarity Encoder" for dirty categories](https://www.slideshare.net/GaelVaroquaux/machine-learning-on-non-curated-data-154905090).
# 
# However, it seems like dirty_cat doesn't handle missing values or unknown categories as well as category_encoders does. And you may need to use it with one column at a time, instead of with your whole dataframe.
# 
# **4. [Embeddings](https://www.kaggle.com/colinmorris/embedding-layers)** can work well with sparse / high cardinality categoricals.
# 
# _**I hope it’s not too frustrating or confusing that there’s not one “canonical” way to encode categoricals. It’s an active area of research and experimentation — maybe you can make your own contributions!**_
# %% [markdown]
# ### Setup
# 
# You can work locally (follow the [local setup instructions](https://lambdaschool.github.io/ds/unit2/local/)) or on Colab (run the code cell below).

# %%
get_ipython().run_cell_magic('capture', '', "import sys\n\n# If you're on Colab:\nif 'google.colab' in sys.modules:\n    DATA_PATH = 'https://raw.githubusercontent.com/LambdaSchool/DS-Unit-2-Kaggle-Challenge/master/data/'\n    !pip install category_encoders==2.*\n\n# If you're working locally:\nelse:\n    DATA_PATH = '../data/'")


# %%
DATA_PATH = 'https://raw.githubusercontent.com/LambdaSchool/DS-Unit-2-Kaggle-Challenge/master/data/'
import pandas as pd
from sklearn.model_selection import train_test_split

train = pd.merge(pd.read_csv(DATA_PATH+'waterpumps/train_features.csv'), 
                 pd.read_csv(DATA_PATH+'waterpumps/train_labels.csv'))
test = pd.read_csv(DATA_PATH+'waterpumps/test_features.csv')
sample_submission = pd.read_csv(DATA_PATH+'waterpumps/sample_submission.csv')

train.shape, test.shape


# %%
train

# %% [markdown]
# ## Train test split

# %%
from sklearn.model_selection import train_test_split

# Splitting the feature matrix in train and validation for later testing
train,val = train_test_split(train, test_size = 0.2, random_state = 42)


# %%
val

# %% [markdown]
# ## EDA

# %%
# !pip install pandas_profiling==2.*


# %%
# from pandas_profiling import ProfileReport
# profile = ProfileReport(train, minimal=True).to_notebook_iframe()
# profile


# %%
import plotly
print(plotly.__version__)


# %%
# !pip install plotly==4.4.1


# %%
import plotly.express as px
fig = px.scatter(train,x=train['longitude'], y=train['latitude'])
fig.show()


# %%
condition = train['longitude'] == 0
train[condition]


# %%
# Cleaning the outliers and feature engineering
import numpy as np

def wrangle(X):
  X = X.copy()
  X['latitude'] = X['latitude'].replace(-2e-08, 0)
  cols_with_zeros = ['longitude', 'latitude', 'construction_year', 'gps_height', 'population']
  for col in cols_with_zeros:
    X[col] = X[col].replace(0, np.nan)
    X[col+'_MISSING'] = X[col].isnull()
    # drop the duplicate columns
  X.drop(['quantity_group', 'quantity'], axis=1, inplace=True)

  # Drop recorded by (never varies) and id(always varies, random)
  X.drop(['recorded_by', 'id'], axis=1, inplace=True)

  #convert date into datetime object
  X['date_recorded'] = pd.to_datetime(X['date_recorded'], infer_datetime_format=True)

  #Extract components from date_recorded then drop the original column
  X['year_recorded'] = X['date_recorded'].dt.year
  X['month_recorded'] = X['date_recorded'].dt.month
  X['day_recorded'] = X['date_recorded'].dt.day
  X.drop('date_recorded', axis=1, inplace=True)

  #Engineer the features: how many years from construction_year to date_recorded
  X['year'] = X['year_recorded'] - X['construction_year']
  X['year_MISSING'] = X['year'].isnull()

  #return the wrangled dataframe
  return X

train = wrangle(train)
val = wrangle(val)
test = wrangle(test)
train

# %% [markdown]
# ## target and features selection

# %%
# target and features selection
target = 'status_group'

# get a dataframe with all train columns except the target
train_features = train.drop(columns=[target])

# get a list of numeric features
numeric_features = train_features.select_dtypes(include='number').columns.tolist()

#get a series with the cardinality of the non numeric function
cardinality = train_features.select_dtypes(exclude='number').nunique()

#get a lit of all cardinal features with carinality <= 50
categorical_features = cardinality[cardinality <= 50].index.tolist()

#combine the lists
features = numeric_features + categorical_features




# %%
#arrange the data into X features and y target vector

X_train = train[features]
y_train = train[target]
X_val = val[features]
y_val = val[target]
X_test = test[features]

# %%

# %%
X_train.info()

# %% [markdown]
# ## Using Random Forest for fitting by creating a pipeline

# %%
import category_encoders as ce
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(
    ce.OneHotEncoder(use_cat_names=True), 
    SimpleImputer(strategy='median'),
    RandomForestClassifier(random_state=0, n_jobs=-1)
)

# fit on train, score on val
pipeline.fit(train,train)
pipeline.score(X_val,y_val)


# %%



# %%



