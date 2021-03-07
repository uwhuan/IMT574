# Stayzilla Analysis

This repo contains source code for the final project for IMT 574

## 0. Data Preprocessing

### Cleaning Steps

* Drop trivia columns such as the scrap time, check in date, check out date, etc.
* Clean up `room_price` column to get rid of “per” string
* Split `Additional Info` column to separate response time and acceptance rate



## 1. Service Value Prediction

### Use Case

* As a member of marketing team in Stayzilla Co. I want to know whether we need to promote our service (verify their property with us) to the owner of a property based on its current data shared with us. 
* As an owner of a property that has been registered in Stayzilla, I want to know whether I should verify my property by following the choice of other owners whose properties are similar to mine.
* As a customer who wants to stay in one of the property in Stayzilla, I want to know how much I can trust a non-verified property by learning the service value of similar properties.

### Solution: Use a k-nearest Neighbors 

#### Rational

Based on the requirement, we want to use a binary classifier based on the numerical and boolean features. Here we use k-nearest neighbors algorithm. This strategy has several advantages on our requirement: first, it has no assumptions for inputs, which saves us lots of work since we have lots of input features; second, it can evolve constantly with new data; third, it has only one hyper parameter which is easy to implement.

#### [Implementation Details](https://github.com/uwhuan/IMT574/blob/master/knn_classify.py)

We first uses all the numeriacl and boolean features from our cleaned dataset  as independent variables, and use the `service value` column as the predictive variable. Then, we build a knn model with iterative choice of $k$. For each iteration, we calculate the accuracy and f1 score of the model by 20 times and uses the average score. We tried $k$ in range from 0 to 300 and find the best $k$ neighbors.

### Results

The following is a plot of the accuracy and f1 scores. We can see that it can achieve high accuracy and f1 score in some k despite of some variation. 

 ## 2. Clustering

### Use Case

* As a customer of Stayzilla, I want to know the similar properties based on my current choice. 
* As an software engineer of Stayzilla, I want to implement a function that when the user is viewing a page of certain property, I can recommend similar properties who offer similar features (amenities, location, service, etc).
* As an owner of a property that has been registered in Stayzilla, I want to know what are the other properties that offers similar services so that I can make ajustment to my current marketing strategy.

### Solution: K-means Clustering

#### Rational

Based on the requirement, we want to use an unsupervised learning model to cluster the properties into several groups and plot the results on the map, so that the customers and the property owners can find the similar properties nearby. Here we choose K-means clustering because it scales well to large data sets, easily adapts to new data and guarantees convergence. 

#### [Implementation Details](https://github.com/uwhuan/IMT574/blob/master/kmeans_cluster.py)

We first uses all the numeriacl and boolean features from our cleaned dataset  as independent variables. Then we iteratively build a kmeans model using cluster amount from 4 to 15. We want to choose as small clustering as possible to give less confusion for the stakeholders, and keeps the reliability at the same time. We use the inertia value to evaluate our model since it calculates the SSE from each point to the centroid. Finally, we choose 6 as our cluster amounts and plot our results on the map.

### Results

Cluster map.  Cluster and price map

The color indicates the different clusterings for each property. We can see that the red and orange clusters dominates most of the places. However, when we add the price feature, we can see that the green cluster has high average prices.

 ## 4. Expectation Maximization

### Use Case

* As a owner of the properties, I want to know whether there are any similarities regarding the prices of the properties. If so, I want to set prices for my properties according to the properties that possess similar features as my properties.

### Solution: Gaussian mixture model

#### Rational

When thinking about this question, we wonder whether the properties cluster in regards of prices, and if so, how would they cluster. We first pick several variables which we believe might be the attributes of the prices; then, we explore the clustering with Expectation Maximization (EM). It is because, in this case, to our knowledge, there is not any clear separation between the clustering. In case like this, the reality of the situation is that there will be uncertainty about the cluster assignments, so it is ideal to use an approach that reflects that. Specifically, we are using a finite Gaussian mixture model (note that the EM algorithm is just the way you estimate the GMM, it isn't the clustering model itself) is one way to respect that fact about this particular situation.

The features we believe might determine prices of the properties are the description length, the type of the rooms (whether it is deluxe or not), number of amenities, if the properties have wifi, if the properties have air conditioner, if the properties serve breakfast, service value, occupancy of adults and of children. We want to use these features to explore how all the properties are related to each other in regards of the prices.

Since we do not have established knowledge on what would be the optimal numbers of the clusters, I first intended to try different clusters number to explore how the AIC and BIC change with the clusters number. Also, I found that when using 90 as the max iteration number, the AIC and BIC results are relatively the smallest. Based on these thoughts and observations, I built up a model looping from 1 to 25 as clusters number, run 25 times with 100 iterations each time, and return the average AIC and BIC values for each cluster numbers. To further explore, I also built up two models using 1 to 15 and 1 to 10 as the cluster numbers.

### Results

AIC and BIC values changing plots with different ranges of clusters numbers. 