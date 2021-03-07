#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 03:39:12 2021

@author: lizzychen
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture

data = pd.read_csv('stayzillaCleanFeatures.csv',index_col=0)

housing = data.loc[:,['descr_len','deluxe','num_amenities','wifi','ac','breakfast','service_value','adult_occupancy','child_occupancy']]
housing = housing.fillna(value=housing.mean())

average_aic = 0;
average_bic = 0;
aic_change = []
bic_change = []
cluster_num = []

for components_num in range(1, 11):
    cluster_num.append(components_num)
    sum_aic = 0;
    sum_bic = 0;
    for r in range(1, 21):
        model = GaussianMixture(n_components = components_num, init_params='random', max_iter=100)
        model.fit(housing)
        sum_aic += model.aic(housing)
        sum_bic += model.bic(housing)
    
    average_aic = sum_aic / 20
    average_bic = sum_bic / 20
    aic_change.append(average_aic)
    bic_change.append(average_bic)
    
yhat = model.predict(housing)
print(yhat)

print(average_aic)
print(average_bic)

plt.plot(cluster_num, aic_change, marker='', color='blue', linewidth=2, linestyle='dashed', label="aic")
plt.plot(cluster_num, bic_change, marker='', color='olive', linewidth=2, linestyle='dashed', label="bic")
plt.xlabel("Number of Clusters")
plt.ylabel("Measurement")
plt.legend()
plt.show()
