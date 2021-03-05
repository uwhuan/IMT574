import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
import folium
import matplotlib.pyplot as plt

df = pd.read_csv('./data/stayzilla_rough_cleaned.csv')

# Make a copy for analysis
df_work = df.copy()

# Select the input features
X = df_work[['latitude', 'longitude','room_price', 'Newspaper', 'Parking','service_value',
       'Card Payment', 'Elevator', 'Pickup & Drop', 'Veg Only', 'Bar', 'Laundry', 'adult_occupancy',
       'child_occupancy', 'num_amenities', 'zones', 'descr_len', 'deluxe', 'num_simhotel', 'wifi', 'ac', 'breakfast',
       'response_label', 'Apartment', 'Homestay', 'Hotel', 'House', 'Lodge', 'Resort', 'Spa', 'Villa']]

# Fit into classifier to determine the best number of clusters
def km(X, n):
    model = KMeans(n_clusters=n).fit(X)
    # According to the documentation, the inertia calculate the SSE between the point and the centroid
    # We can use it to evaluate our model
    iner = model.inertia_
    pred = model.predict(X)
    return iner, pred

labels = []
inertia = []
list_k = list(range(4, 15))
for n in list_k:
    a, b = km(X, n)
    inertia.append(a)
    labels.append(b)

# Using sse to find the best number of clustering
plt.figure(figsize=(6, 6))
plt.plot(list_k, inertia, '-o')

# The line drops significantly from 4 to 6, 
# since we don't want too much clustering to confuse stakeholders, 
# we will choose 6 in our project
df_temp = pd.DataFrame(labels[2], columns=['class'])

# Combine the cluster results with other information for visualization
df_map = df_work[['longitude', 'latitude', 'room_price', 'property_name']]
df_map = pd.concat([df_map, df_temp], axis=1)

# Plot each property on the map, and use different color to indicate the clustering
m = folium.Map(
    location=[21.15, 79.09],
    zoom_start=4
)

colors = ['red', 'brown', 'green', 'blue', 'purple', 'orange']
for i in range(0, 7):
    df_map[df_map['class'] == i].apply(lambda ll: folium.Circle(radius=ll.room_price,
                                              location=[ll.latitude, ll.longitude],
                                              stroke = True,
                                              color=colors[i],
                                              opacity=0.3,
                                              popup=ll.property_name).add_to(m), axis='columns')

m.save("./res/cluster.html")

# The map above is good to see the price, but difficult to see exact location
# The code below will plot another map with fix radius
m = folium.Map(
    location=[21.15, 79.09],
    zoom_start=4
)

for i in range(0, 7):
    df_map[df_map['class'] == i].apply(lambda ll: folium.Circle(radius=200,
                                              location=[ll.latitude, ll.longitude],
                                              stroke = True,
                                              fill=True,
                                              color=colors[i],
                                              popup=ll.property_name).add_to(m), axis='columns')

m.save("./res/cluster_clear.html")

# Show the room price distribution in each class
image = df_map['room_price'].hist(by=df_map['class'], figsize=(16, 7))
plt.savefig('./res/kmeans_class_price.png', bbox_inches='tight')