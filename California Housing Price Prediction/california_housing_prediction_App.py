#!/usr/bin/env python
# coding: utf-8

# In[308]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import shap
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')


st.write("""
# California Housing Price Prediction

This app predicts the **California housing price** for house distribution per blocks
""")
# In[309]:


@st.cache(allow_output_mutation=True)
def load_data():

    california_df = pd.read_excel('1553768847_housing.xlsx')
    return california_df


# load data
california_df = pd.read_csv('california_housing.csv')

# In[310]:


# st.write("California Housing Price Dataset has {} samples and {} features each.".format(
#    *california_df.shape))

# In[312]:


# In[313]:


california_df['total_bedrooms'].fillna(
    california_df['total_bedrooms'].mean(), inplace=True)


# In[314]:
st.sidebar.header("Specify Input Parameters")


def user_input():
    ocean_proximity = st.sidebar.selectbox(
        "ocean proximity", california_df['ocean_proximity'].unique())
    longitude = st.sidebar.slider("longitude", -124.35, -114.31, -119.56)
    latitude = st.sidebar.slider("latitude", 32.5, 41.95, 35.63)
    housing_median_age = st.sidebar.slider("housing median age", 1, 52, 29)
    total_rooms = st.sidebar.slider("total rooms", 2, 39320, 2635)
    total_bedrooms = st.sidebar.slider("total bedrooms", 1.0, 6445.0, 537.87)
    population = st.sidebar.slider("population", 3, 35682, 1425)
    households = st.sidebar.slider("households", 1, 6082, 500)
    median_income = st.sidebar.slider("median income", 0.4999, 15.0001, 3.8706)
    data = {'longitude': longitude,
            'latitude': latitude,
            'housing_median_age': housing_median_age,
            'total_rooms': total_rooms,
            'total_bedrooms': total_bedrooms,
            'population': population,
            'households': households,
            'median_income': median_income,
            'ocean_proximity': ocean_proximity}
    features = pd.DataFrame(data, index=[0])
    return features


input_df = user_input()


california = california_df.drop('median_house_value', axis=1)
california = pd.concat([input_df, california], axis=0)


# Encode columnn
#le = LabelEncoder()
#dummy = le.fit_transforn(california['ocean_proximity'])
#dummy = pd.DataFrame(dummy)
#dummy = dummy.rename(columns={dummy.ocean_proximity[0]:'ocean_proximity_le'})

dummy = pd.get_dummies(
    california['ocean_proximity'], prefix='ocean_parameters')

california = pd.concat([california, dummy], axis=1)
california = california.drop('ocean_proximity', axis=1)
# select only the first row that is input data
california_input = california[:1]
# train the dataset
california_train = california[1:]
st.subheader("User Input Features")

st.write(california_input)
st.write('---')
# In[315]:


# # Dataset Description :
# # -Variable and Description
# ●	longitude (signed numeric - float) : Longitude value for the block in California, USA
# ●	latitude (numeric - float ) : Latitude value for the block in California, USA
# ●	housing_median_age (numeric - int ) : Median age of the house in the block
# ●	total_rooms (numeric - int ) : Count of the total number of rooms (excluding bedrooms) in all houses in the block
# ●	total_bedrooms (numeric - float ) : Count of the total number of bedrooms in all houses in the block
# ●	population (numeric - int ) : Count of the total number of population in the block
# ●	households (numeric - int ) : Count of the total number of households in the block
# ●	median_income (numeric - float ) : Median of the total household income of all the houses in the block
# ●	ocean_proximity (numeric - categorical ) : Type of the landscape of the block
# [ Unique Values : 'NEAR BAY', '<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'ISLAND'  ]
# ●	median_house_value (numeric - int ) : Median of the household prices of all the houses in the block
#

# In[316]:

x = california_train
y = california_df['median_house_value']


# In[317]:


# In[318]:


# In[319]:


sc = StandardScaler()
x_sc = sc.fit_transform(x)


rfmodel = RandomForestRegressor()
rfmodel.fit(x, y)

# apply model to make prediction
rfmodel_pred = rfmodel.predict(california_input)

# Finally make use of Random forest model as it is the best from the metrics to make prediction in the app
st.write("# Prediction of housing median values")
st.write(rfmodel_pred)
st.write('---')
# In[328]:


# explaining the model's prediction using shap values
explainer = shap.TreeExplainer(rfmodel)
shap_values = explainer.shap_values(x)

st.header('Feature Importance')
plt.title('Feature Importance based on Shap Values')
shap.summary_plot(shap_values, x)
st.pyplot(bbox_inches='tight')
st.write('---')

plt.title('Feature Importance based on Shap Values (bar chart)')
shap.summary_plot(shap_values, x, plot_type='bar')
st.pyplot(bbox_inches='tight')


#
# # Random Forest Regression Model is the better model when prediction median house value using all independent variables, so we can use it as the final model
