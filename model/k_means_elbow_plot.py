import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# Load data
df = pd.read_csv('../data/Iris.csv')
df.drop(labels='Id', axis=1, inplace=True) # Remove Id column

# Normalize the data
input_label = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
data = df[input_label].to_numpy()

scaler = MinMaxScaler()
scaler.fit(data)
data = scaler.transform(data)

for ind, label in enumerate(input_label):
	df[label] = data[:, ind]

# Iterating through different K's
see = []
k_list = [k for k in range(1,11)]
for k in k_list:
	km = KMeans(n_clusters=k)
	km.fit(df[input_label])

	see.append(km.inertia_)

# Making SSE plot
plt.plot(k_list, see)
plt.xticks(k_list)
plt.ylabel('SSE')
plt.xlabel('K')
plt.grid()
plt.title('Elbow Plot for K-Means Clustering of Iris Data')
plt.savefig('../gfx/elbowPlot.png', dpi=800)
plt.clf()

# Making Normalized SEE plot
high = max(see)
low = min(see)
see_normal = []
for error in see:
	see_normal.append(error/(high-low))

plt.plot(k_list, see_normal)
plt.xticks(k_list)
plt.ylabel('SEE')
plt.xlabel('K')
plt.grid()
plt.title('Elbow Plot for K-Means Clustering of Iris Data (Normalized)')
plt.savefig('../gfx/elbowPlot_normalized.png', dpi=800)