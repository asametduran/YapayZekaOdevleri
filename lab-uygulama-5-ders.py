
import pandas as pd 
import simpsom as sps
from sklearn.cluster import KMeans
import numpy as np

veri = pd.read_csv("C:/Users/burak/Desktop/Uygulama 5.1/airline-safety.csv")
X = veri.drop(["airline","avail_seat_km_per_week"],axis=1)


net = sps.SOMNet (20, 20, X.values, PBC=True)


net.train (0.01, 10000)

hrt = np.array((net.project(X.values))) 
kmeans = KMeans(n_clusters = 3, max_iter
=
300, random_state
=
0)

y_kmeans = kmeans.fit_predict(hrt)

veri["kümeler"] = kmeans.labels_
print(veri[veri["kümeler"]==0].head(5))
print(veri[veri["kümeler"]==1].head(5))
print(veri[veri["kümeler"]==2].head(5))