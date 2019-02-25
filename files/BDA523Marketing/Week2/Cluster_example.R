  # kmeans(x, centers, iter.max = 10, nstart = 1,
#        algorithm = c("Hartigan-Wong", "Lloyd", "Forgy",
#                      "MacQueen"), trace=FALSE)

# X is your data frame or matrix.  All values must be numeric.
# If you have an ID field make sure you drop it or it will be included as part of the centroids.
# Centers is the K of K Means.  centers = 5 would results in 5 clusters being created.
# You have to determine the appropriate number for K.
# iter.max is the number of times the algorithm will repeat the cluster assignment and moving of centroids.
# nstart is the number of times the initial starting points are re-sampled.
# In the code, it looks for the initial starting points that have the lowest within sum of squares (withinss).
# That means it tries “nstart” samples, does the cluster assignment for each data point “nstart” times, and picks the centers that have the lowest distance from the data points to the centroids.
# trace gives a verbose output showing the progress of the algorithm.

data <-read.csv("C:/Kalender/MEF&SYLABUS/Clustering/Wholesale customers data.csv")
summary(data)

top.n.custs <- function (data,cols,n=5) { #Requires some data frame and the top N to remove
  idx.to.remove <-integer(0) #Initialize a vector to hold customers being removed
  for (c in cols){ # For every column in the data we passed to this function
    col.order <-order(data[,c],decreasing=T) #Sort column "c" in descending order (bigger on top)
    #Order returns the sorted index (e.g. row 15, 3, 7, 1, ...) rather than the actual values sorted.
    idx <-head(col.order, n) #Take the first n of the sorted column C to
    idx.to.remove <-union(idx.to.remove,idx) #Combine and de-duplicate the row ids that need to be removed
  }
  return(idx.to.remove) #Return the indexes of customers to be removed
}
top.custs <-top.n.custs(data,cols=3:8,n=5)
length(top.custs) #How Many Customers to be Removed?
data[top.custs,] #Examine the customers
data.rm.top<-data[-c(top.custs),] #Remove the Customers


###### Alternative Use Remove Outliers #####


remove_outliers <- function(x, na.rm = TRUE, ...) {
  qnt <- quantile(x, probs=c(.25, .75), na.rm = na.rm, ...)
  H <- 1.5 * IQR(x, na.rm = na.rm)
  y <- x
  y[x < (qnt[1] - H)] <- NA
  y[x > (qnt[2] + H)] <- NA
  y
}

data$Fresh<- remove_outliers(data$Fresh)
data$Milk<- remove_outliers(data$Milk)
data$Grocery<- remove_outliers(data$Grocery)
data$Frozen<- remove_outliers(data$Frozen)
data$Detergents_Paper<- remove_outliers(data$Detergents_Paper)
data$Delicassen<- remove_outliers(data$Delicassen)


data.rm.top <- data[complete.cases(data), ]



### Now, using data.rm.top, we can perform the cluster analysis.  Important note: We’ll still need to drop the Channel and Region variables.  These are two ID fields and are not useful in clustering.


set.seed(1234) #Set the seed for reproducibility
k <-kmeans(data.rm.top[,-c(1,2)], centers=5) #Create 5 clusters, Remove columns 1 and 2
k$centers #Display&nbsp;cluster centers
table(k$cluster) #Give a count of data points in each cluster


rng<-2:20 #K from 2 to 20
tries <-100 #Run the K Means algorithm 100 times
avg.totw.ss <-integer(length(rng)) #Set up an empty vector to hold all of points
for(v in rng){ # For each value of the range variable
  v.totw.ss <-integer(tries) #Set up an empty vector to hold the 100 tries
  for(i in 1:tries){
    k.temp <-kmeans(data.rm.top,centers=v) #Run kmeans
    v.totw.ss[i] <-k.temp$tot.withinss#Store the total withinss
  }
  avg.totw.ss[v-1] <-mean(v.totw.ss) #Average the 100 total withinss
}
plot(rng,avg.totw.ss,type="b", main="Total Within SS by Various K",
     ylab="Average Total Within Sum of Squares",
     xlab="Value of K")


library(cluster)
clusplot(data.rm.top, k$cluster, color=TRUE, shade=TRUE, 
         labels=4, lines=0, main="K-means cluster plot")

#### Hierarchical Clustering ####


# Step 01- obtain distance matrix (right way)

h_data <- data.rm.top[,-c(1,2)]

my_dist <- dist(h_data, method = "euclidean")


# Step 02- Apply Hierarchical Clustering
fit <- hclust(my_dist, method="ward.D2")


# Step 03- Display dendogram
plot(fit)


Dendogram_Height=0
for (i in 2:15) Dendogram_Height[i] <- fit$height[i-1]
plot(1:15, Dendogram_Height, type="b", xlab="Sequence of merging",
     ylab="Dendogram Height")
plot(15:1, Dendogram_Height, type="b", xlab="# of clusters",
     ylab="Dendogram Height")


# Step 04- draw dendogram with color borders 
# One can use this step to take a look at execution
rect.hclust(fit, k=8, border="red")
plot(fit)
rect.hclust(fit, k=7, border="red")
plot(fit)
rect.hclust(fit, k=6, border="red")


# draw color borders around required clusterd
plot(fit)
rect.hclust(fit, k=7, border="blue")


my_groups <- cutree(fit, k=8)

###  Cluster Assgnments ###

my_last_df <- data.frame(h_data,my_groups)

aggregate(my_last_df,by=list(my_last_df$my_groups),FUN=mean) # get cluster means



