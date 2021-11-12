#!/usr/bin/env python
# coding: utf-8

# **Author: Fitria Dwi Wulandari (wulan391@sci.ui.ac.id) - December 24, 2020.**

# # Data Cleansing

# **Problem**: many customers are switching subscriptions to competitors so that the management wants to reduce the number of customers who switch (churn) by using machine learning.
# 
# **Goal**: perform data preprocessing in June 2020.
# 
# The steps to be taken are as follows:
# 1. Validating the customer ID number
# 2. Handling missing values
# 3. Handling outlier
# 4. Standardize the value of the variable

# ### Import Dataset

# In[3]:


# Import libraries
import pandas as pd
pd.options.display.max_columns = 50


# In[4]:


# Import dataset
telco = pd.read_csv('dqlab_telco.csv')
print('Dataset size: %d columns dan %d rows.\n' % telco.shape)
telco.head()


# - `UpdatedAt` : Periode of Data taken
# - `customerID` : Customer ID
# - `gender` : Whether the customer is a male or a female (Male, Female)
# - `SeniorCitizen` : Whether the customer is a senior citizen or not (1, 0)
# - `Partner` : Whether the customer has a partner or not (Yes, No)
# - `Dependents` : Whether the customer has dependents or not (Yes, No)
# - `tenure` : Number of months the customer has stayed with the company
# - `PhoneService` : Whether the customer has a phone service or not (Yes, No)
# - `MultipleLines` : Whether the customer has multiple lines or not (Yes, No, No phone service)
# - `InternetService` : Customer’s internet service provider (DSL, Fiber optic, No)
# - `OnlineSecurity` : Whether the customer has online security or not (Yes, No, No internet service)
# - `OnlineBackup` : Whether the customer has online backup or not (Yes, No, No internet service)
# - `DeviceProtection` : Whether the customer has device protection or not (Yes, No, No internet service)
# - `TechSupport` : Whether the customer has tech support or not (Yes, No, No internet service)
# - `StreamingTV` : Whether the customer has streaming TV or not (Yes, No, No internet service)
# - `StreamingMovies` : Whether the customer has streaming movies or not (Yes, No, No internet service)
# - `Contract` : The contract term of the customer (Month-to-month, One year, Two year)
# - `PaperlessBilling` : Whether the customer has paperless billing or not (Yes, No)
# - `PaymentMethod` : The customer’s payment method (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic))
# - `MonthlyCharges` : The amount charged to the customer monthly
# - `TotalCharges` : The total amount charged to the customer
# - `Churn`  : The customer churned or not (Yes or No)

# In[6]:


# Number of unique IDs
print('The number of Unique IDs is', telco.customerID.nunique())


# ### Validating the Customer ID Number

# #### Filtering Customer ID Number with Specific Format

# Search the correct customer ID number format (phone number), with the following criteria:
# - The length of a character is 11-12.
# - Consists of numbers only.
# - The first 2 digits of customerID is 45

# In[7]:


telco['valid_id'] = telco['customerID'].astype(str).str.match(r'(45\d{9,10})')
telco = (telco[telco['valid_id'] == True]).drop('valid_id', axis = 1)


# In[8]:


print('The number of filtered Customer ID is',telco['customerID'].count())


# #### Filtering Duplicate Customer ID Number

# Ensure that there are no duplicate ID. The type of duplication is:
# - Duplication due to inserting exceeds once with the same value for every column
# - Duplication due to inserting different data collection periods

# In[11]:


print('Dataset size: %d columns dan %d rows.\n' % telco.shape)
print('The number of duplicate data is', telco.duplicated().sum())


# In[12]:


# Drop duplicate rows
telco.drop_duplicates()

# Drop duplicate customerID sorted by UpdatedAt
telco = telco.sort_values('UpdatedAt', ascending=False).drop_duplicates(['customerID'])
print('The number of distinct customer ID is',telco['customerID'].count())


# #### Summary

# The validity of the customer ID number is very necessary to ensure that the data we have collected is correct. Based on these results, there are differences in the number of ID numbers from the first data loaded up to the final results. The number of data rows when first loaded was 7113 rows and 22 columns with 7017 unique ID. Then after checking the validity of the customer ID, there are only 6993 rows of data left.

# ### Handling Missing Values

# #### Detecting Missing Values

# In[23]:


print('Status Missing Values :',telco.isnull().values.any())
print('\nThe number of Missing Values for each columns:')
print(telco.isnull().sum().sort_values(ascending=False))


# After further analysis, it turns out there are still Missing Values from the data that we have validated the Customer ID Number. Missing values are in the `Churn`, `tenure`, `MonthlyCharges` & `TotalCharges` columns. 

# #### Handling Missing Values

# Next, we will eleminate rows from data that are not detected whether churn or not. We only accepts data that has the churn flag or not.

# In[13]:


print('The number of missing values from the Churn column is',telco['Churn'].isnull().sum())


# In[14]:


# Dropping all rows with spesific column (churn)
telco.dropna(subset=['Churn'],inplace=True)
print('The number of rows and columns after deleting data with missing value is',telco.shape)


# #### Handling Missing Values with Imputation

# We will fill the missing value with following criteria:
# - Missing values Tenure fill with 11
# - Missing values numeric variable except Tenure fill with median of the non-missing values in each columns

# In[16]:


# Handling missing values Tenure fill with 11
telco['tenure'].fillna(11, inplace=True)

# Handling missing values numeric vars (except Tenure)
for col_name in list(['MonthlyCharges','TotalCharges']):
    median = telco[col_name].median()
    telco[col_name].fillna(median, inplace=True)
print('\nThe number of Missing Value after imputation:')
print(telco.isnull().sum().sort_values(ascending=False))


# After we handle by eliminating rows and filling rows with certain values, it is proven that there are no missing values in the data anymore, as evidenced by the number of missing values for each variable that is 0.

# ### Handling Outliers

# #### Detecting Outlier with Boxplot

# The most common graphical ways of detecting outliers is the boxplot. Box plot is a method for graphically depicting groups of numerical data through
# their quartiles. Box plots may also have lines extending vertically from the boxes (whiskers) indicating variability outside the upper and lower quartiles.

# In[17]:


print('\nDistribution of data before handled by Outlier: ')
print(telco[['tenure','MonthlyCharges','TotalCharges']].describe())

# Creating Box Plot
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure()
sns.boxplot(x=telco['tenure'])
plt.show()

plt.figure()
sns.boxplot(x=telco['MonthlyCharges'])
plt.show()

plt.figure()
sns.boxplot(x=telco['TotalCharges'])
plt.show()


# From the three boxplots with the variable `tenure`, `MonthlyCharges` & `TotalCharges` we can look that there are outliers. This can be identified from the
# points that far away from the boxplot.

# #### Handling Outlier

# After we know which variables that have outliers, handle it by changing that value to the Maximum & Minimum value of the interquartile range (IQR).

# In[18]:


# Handling with IQR
Q1 = (telco[['tenure','MonthlyCharges','TotalCharges']]).quantile(0.25)
Q3 = (telco[['tenure','MonthlyCharges','TotalCharges']]).quantile(0.75)
IQR = Q3 - Q1

maximum = Q3 + (1.5*IQR)
print('Maximum value for each columns is: ')
print(maximum)
minimum = Q1 - (1.5*IQR)
print('\nMinimum value for each columns is: ')
print(minimum)

more_than = (telco > maximum)
lower_than = (telco < minimum)
telco = telco.mask(more_than, maximum, axis=1)
telco = telco.mask(lower_than, minimum, axis=1)
print('\nDistribution of data after handled by Outlier:: ')
print(telco[['tenure','MonthlyCharges','TotalCharges']].describe())


# After handling the outliers, and looking at the details of the data, there are no outlier values.

# ### Standardize Values

# #### Detecting non standard value

# Detecting whether there are values from non-standard categorical variables. Usually this cases occurs because data entry mistakes. The difference of
# term is one of the factors, so we need to standardize the data.

# In[19]:


for col_name in list(['gender','SeniorCitizen','Partner','Dependents','PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaperlessBilling','PaymentMethod','Churn']):
    print('\nUnique Values Count \033[1m' + 'Before Standardized \033[0m Variable',col_name)
    print(telco[col_name].value_counts())


# We can see that there are some non-standard variables. These variables are:
# - Gender (Female, Male, Wanita, Laki-Laki), which can be standardized become (Female, Male).
# - Dependents (Yes, No, Iya), can be standardized become(Yes, No).
# - Churn (Yes, No, Churn), can be standardized become(Yes, No).

# #### Standardized categorical variable

# After we know which variables that have non-standard values, then we have to standardize it with the most values term, without changing the meaning.
# Example: Iya -> Yes. Then look again the unique values of each variable that has been changed.

# In[20]:


telco = telco.replace(['Wanita','Laki-Laki','Churn','Iya'],['Female','Male','Yes','Yes'])

for col_name in list(['gender','Dependents','Churn']):
    print('\nUnique Values Count \033[1m' + 'After Standardized \033[0mVariable',col_name)
    print(telco[col_name].value_counts())


# After we standardize the values, now dataset ready to analyze.
