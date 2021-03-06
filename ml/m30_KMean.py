from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd

datasets = load_iris()

irisDF = pd.DataFrame(data=datasets.data, columns=datasets.feature_names)
print(irisDF)

kmean = KMeans(n_clusters=3, max_iter=300, random_state=66)
kmean.fit(irisDF)

results = kmean.labels_
# print(results)
# print(datasets.target) # 원래 y값

irisDF['cluster'] = kmean.labels_ # 클러스터링 해서 생성한 y값
irisDF['target'] = datasets.target # 원래 y 값

# print(datasets.feature_names)
# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
iris_results = irisDF.groupby(['target', 'cluster'])['sepal length (cm)'].count()
# iris_results = irisDF.groupby(['target', 'cluster']).count()
print(iris_results)