import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# Loading data
df = pd.read_csv('../data/Iris.csv')
df.drop(labels='Id', axis=1, inplace=True)

## === Pre Processing === ###
input_label = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
data = df[input_label].to_numpy()

# Normalizing data
scaler = MinMaxScaler()
scaler.fit(data)
data = scaler.transform(data)

for ind, label in enumerate(input_label):
	df[label] = data[:, ind]

print(df.head())

k = 3
km = KMeans(n_clusters=k)
y_predicted = km.fit_predict(df[input_label])
print(km.labels_)

# Grouping the results
df['cluster'] = y_predicted

df1 = df[df.cluster==0]
df2 = df[df.cluster==1]
df3 = df[df.cluster==2]

# plotting the results
plt.scatter(df1['SepalLengthCm'], df1['SepalWidthCm'], color='green')
plt.scatter(df2['SepalLengthCm'], df2['SepalWidthCm'], color='blue')
plt.scatter(df3['SepalLengthCm'], df3['SepalWidthCm'], color='black')

plt.xlabel('SepalLengthCm (Normalized)')
plt.ylabel('SepalWidthCm (Normalized)')
plt.title(f'K-Means Clustering Results for k={k}')
plt.savefig(f'../gfx/K-Means_clutering_k{k}.png', dpi=800)
plt.clf()

# Plot original
df1 = df[df.Species == 'Iris-setosa']
df2 = df[df.Species == 'Iris-versicolor']
df3 = df[df.Species == 'Iris-virginica']

plt.scatter(df1['SepalLengthCm'], df1['SepalWidthCm'], color='blue')
plt.scatter(df2['SepalLengthCm'], df2['SepalWidthCm'], color='green')
plt.scatter(df3['SepalLengthCm'], df3['SepalWidthCm'], color='black')

plt.legend(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
plt.xlabel('SepalLengthCm (Normalized)')
plt.ylabel('SepalWidthCm (Normalized)')
plt.title('Original Data grouping')
plt.savefig('../gfx/K-Means_clustering_original.png', dpi=800)


