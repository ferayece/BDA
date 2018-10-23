import numpy as np
import pandas as pd
from numpy.ma.core import sort
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sb

import pandas as pd
filename = 'directory2.csv'
data = pd.read_csv(filename)
data.info()  #25600
print(data.describe())
data.columns

#describe the categorical columns.
categorical = data.dtypes[data.dtypes == "object"].index
print(data[categorical].describe())

#I dont need all of columns, so delete them:
del data["Postcode"]
del data["Phone Number"]
del data["Street Address"]

#Find unique brands:
categorical = data.dtypes[data.dtypes == "object"].index
print(categorical)
print(data[categorical]["Brand"].describe())  #unique 4, why?
unique_brand = data.groupby('Brand')["Store Number"].nunique()  ##25248 Starbucks
print("Unique Brands:\n",unique_brand)
#data.info()

#Take just Starbucks brand:
import numpy as np
sb_index=np.where(data["Brand"]=="Starbucks")
data=data.loc[sb_index]
print(data.info()) #25249

##Ownership Type
import seaborn as sns
type_colors = ['#78C850',  # Grass
                '#F08030',  # Fire
                '#C03028',  # Electric
                '#E0C068' # Ground
               ]
sns.set(style="whitegrid", context="talk")
sns.countplot(x='Ownership Type', data=data, palette=type_colors)
plt.title("Distribution of Ownership")
plt.show()

#nullity check:
print("Nullity Check:\n",pd.isnull(data).any())
print("Sum of null values for each column:\n",pd.isnull(data).sum())
#Remove the null values from City by row index
missing= np.where(data["City"].isnull() == True)
print(missing[0])
data = data.drop(data.index[missing[0]])
missing= np.where(data["Longitude"].isnull() == True)
data = data.drop(data.index[missing[0]])
missing= np.where(data["Latitude"].isnull() == True)
data = data.drop(data.index[missing[0]])
print("Nullity Check:\n",pd.isnull(data).any())
data.info()

#Group Data according to Countries and Cities and find descriptives:
country_count = data[['Country','City']].groupby(['Country'])['City'] \
                             .count() \
                             .reset_index(name='count') \
                             .sort_values(['count'], ascending=False) #\
print("Country number of stores: ",country_count['count'].count())
print("Average of stores: ",country_count['count'].mean())
print("Max number of stores: ",country_count['count'].max())
print("Min number of stores: ",country_count['count'].min())

#last 10
country_count=country_count.tail(10)
plt.figure(figsize=(10,5))
sns.barplot(country_count['Country'], country_count['count'], alpha=0.8)
plt.title('Last 10 Country')
plt.ylabel('Number of Stores', fontsize=12)
plt.xlabel('Country', fontsize=12)
plt.show()

#first 10
country_count=country_count.head(10)
plt.figure(figsize=(10,5))
sns.barplot(country_count['Country'], country_count['count'], alpha=0.8)
plt.title('Top 10 Country')
plt.ylabel('Number of Stores', fontsize=12)
plt.xlabel('Country', fontsize=12)
plt.show()

# Is Italy in list?
sb_index=np.where(country_count['Country']=="IT")
x=country_count.loc[sb_index]
print(x)

#Group Data according to Cities of Turkey and find descriptives:
tr_data = pd.DataFrame(data.loc[data['Country'] == 'TR'])

city_count = tr_data[['City']].groupby(['City'])['City'] \
                             .count() \
                             .reset_index(name='count') \
                             .sort_values(['count'], ascending=False)
print("Country number of stores: ",city_count['count'].count())
print("Average of stores: ",city_count['count'].mean())
print("Max number of stores: ",city_count['count'].max())
print("Min number of stores: ",city_count['count'].min())

#top 10
city_count=city_count.head(10)
plt.figure(figsize=(10,5))
sns.barplot(city_count['City'], city_count['count'], alpha=0.8)
plt.title('Starbucks in top 10 City of Turkey')
plt.ylabel('Number of Stores', fontsize=12)
plt.xlabel('City', fontsize=12)
plt.show()

#last 10
city_count=city_count.tail(10)
print(city_count)


