# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 19:29:59 2023

@author: laiba
"""

# import necessary libraries
import numpy as np
import pandas as pd
import err_ranges as err
import scipy.optimize as opt
from sklearn.preprocessing import MinMaxScaler
import sklearn.cluster as cluster
import matplotlib.pyplot as plt

def read_files(file1):
    
    """ Reads csv file, converts the csv file to pandas dataframe,
    takes transpose of that dataframe and finally prints the headers 
    of the given data
    """
    
    file1 = pd.read_excel(f"{file1}")
    
    file2 = file1.transpose()

    print("Original Data Frame header")
    print(file1.head)
    print(file1.columns)
    
    return file1, file2

df_climate, df_climate_tr = read_files("climate_change.xlsx")

def clean_data(df):
  """
  This function takes a dataframe as input and removes missing values from each column individually.
  It then returns a balanced dataset with the same number of rows for each column.
  """
  
  CO2_emission = df['CO2 emissions'].dropna()

  agricultural_land = df['Agricultural land '].dropna()

  urban_population = df['Urban population'].dropna()

  min_length = min(len(CO2_emission), len(agricultural_land), len(urban_population))
 
  cluster_data = pd.DataFrame({
        'country': df['Country'].iloc[:min_length].values.tolist(),
        'year': df['Year'].iloc[:min_length].values.tolist(),
        'CO2_emissions': CO2_emission.iloc[:min_length].values.tolist(),
        'agricultural_land': agricultural_land.iloc[:min_length].values.tolist(),
        'urban_population': urban_population.iloc[:min_length].values.tolist()
    })

  
  return cluster_data


cluster_df = clean_data(df_climate)

# Normalizing the data
scaler = MinMaxScaler()

# Selecing columns to normalize
columns_to_normalize = ['CO2_emissions', 'agricultural_land', 'urban_population']

# Normalizing column values
normalized_data = scaler.fit_transform(cluster_df[columns_to_normalize])


# Using KMeans to find clusters in the data
kmeans = cluster.KMeans(n_clusters=4)
kmeans.fit(normalized_data)

cluster_df['cluster'] = kmeans.labels_

# create a scatter plot for agricultural land vs CO2 emissions
for i in range(4):
    clusters = cluster_df[cluster_df['cluster'] == i]
    plt.scatter(clusters['agricultural_land'], clusters['CO2_emissions'], label=f'Cluster {i}')


plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', c='black', label='Cluster Centers')
plt.xlabel('Agricultural Land')
plt.ylabel('CO2 Emissions')
plt.legend()
plt.show()

# create a scatter plot for urban population vs CO2 emissions
for i in range(4):
    clusters = cluster_df[cluster_df['cluster'] == i]
    plt.scatter(clusters['urban_population'], clusters['CO2_emissions'], label=f'Cluster {i}')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', c='black', label='Cluster Centers')
plt.xlabel('urban_population')
plt.ylabel('CO2 Emissions')
plt.legend()
plt.show()

#Analyzing Data for Pakistan

Pak = cluster_df[cluster_df['country'] == 'Pakistan']

# create a plot for pakistan between urban population and CO2 emissions
for i in range(4):
    clusters = Pak[Pak['cluster'] == i]
    plt.scatter(clusters['urban_population'], clusters['CO2_emissions'], label=f'Cluster {i}')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', c='black', label='Cluster Centers')
plt.xlabel('Urban Population')
plt.ylabel('CO2 Emissions')
plt.title('Pakistan')
plt.legend()
plt.show()

#Define the polynomial function
def poly(x, a, b, c):

    y = a*x**2 + b*x + c
    return y

# Data of cluster 2
c1 = cluster_df[(cluster_df['cluster'] == 2)]

# values for curve fitting
w = c1['agricultural_land']
z = c1['CO2_emissions']

popt, pcov = opt.curve_fit(poly, w, z)
sigma = np.sqrt(np.diag(pcov))
lower, upper = err.err_ranges(w, poly, popt,sigma)

# Scatter Plot for cluster 2
plt.plot(w, z, 'o', label='data')
plt.plot(w, poly(w, *popt), '-', label='fit')
plt.fill_between(w, lower, upper, color = "yellow", label='confidence interval')
plt.legend()
plt.xlabel('Agricultural Land')
plt.ylabel('CO2 Emissions')
plt.title("Cluster 2")
plt.show()

# Define the range of future w and z values to make predictions
future_w = np.arange(40, 50)
future_z = poly(future_w, *popt)

# Plot the predictions along with the original data
plt.plot(w, z, 'o', label='data')
plt.plot(w, poly(w, *popt), '-', label='fit')
plt.plot(future_w, future_z, 'o', label='future predictions')
plt.xlabel('Agricultural Land')
plt.ylabel('CO2 Emissions')
plt.title("Cluster 2 Predictions")
plt.legend()
plt.show()