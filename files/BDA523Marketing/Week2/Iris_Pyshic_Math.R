
#-----------------------------------------------
# Hierarchical clustering with the sample data
#------------------------------------------------


# Reading data into R similar to CARDS

temp_str <- "Name physics math
P 15 20
Q 20 15
R 26 21
X 44 52
Y 50 45
Z 57 38
A 80 85
B 90 88
C 98 98"

base_data <- read.table(textConnection(
  temp_str), header = TRUE)
closeAllConnections()

# Check distinct categories of Variables useing STR function
str(base_data)

# Plot data 
plot(base_data$physics, base_data$math, 
     pch=21, bg=c("red","green3","blue","red","green3","blue",
                  "red","green3","blue")[unclass(base_data$Name)],
     main="Base Data")



# Step 01- obtain distance matrix (wrong way)
my_dist <- dist(base_data, method = "euclidean")
# Step 01- obtain distance matrix (right way)
my_dist <- dist(base_data[c(2,3)], method = "euclidean")
print(my_dist)

# Step 02- Apply Hierarchical Clustering
fit <- hclust(my_dist, method="ward.D2")

# Step 03- Display dendogram
# Understand that it shows row number by default
plot(fit)

# change the label to understand it better
plot(fit, labels = base_data$Name)

# Step 04- draw dendogram with color borders 
# One can use this step to take a look at execution
rect.hclust(fit, k=8, border="red")
plot(fit, labels = base_data$Name)
rect.hclust(fit, k=7, border="red")
plot(fit, labels = base_data$Name)
rect.hclust(fit, k=6, border="red")

# draw color borders around required clusterd
plot(fit, labels = base_data$Name)
rect.hclust(fit, k=3, border="blue")

# cut tree into 3 clusters
my_groups <- cutree(fit, k=3)

#------------------------------------------
# Non Hierarchical Clustering
#------------------------------------------


head(iris)
str(iris)
my_iris<-iris
table(my_iris$Species)
km_data<-iris[c(-5)]

# Visualize the iris data

plot(my_iris$Petal.Length, my_iris$Petal.Width, 
     pch=21, bg=c("red","green3","blue")[unclass(my_iris$Species)],
     main="Iris Data" )


#install.packages("plyr")
library("plyr")
ddply(my_iris, "Species", 
      summarise,
      mean_pet_length= mean(Petal.Length),
      mean_pet_Width= mean(Petal.Width)
)


pairs(my_iris[c(1,2,3,4)], main = "Iris Data",pch = 21,  
      bg = c("red","green3","blue")[unclass(my_iris$Species)])

# Manual way
km_fit<- kmeans(km_data, centers=2)
km_fit$withinss
sum(km_fit$withinss)
km_fit<- kmeans(km_data, centers=3)
km_fit$withinss
sum(km_fit$withinss)
km_fit<- kmeans(km_data, centers=4)
km_fit$withinss
sum(km_fit$withinss)
km_fit<- kmeans(km_data, centers=5)
km_fit$withinss
sum(km_fit$withinss)

set.seed(100)
# Step 01- Determine number of clusters - by scree plot
WithinSS <- (nrow(km_data)-1)*sum(apply(km_data,2,var))
for (i in 2:9) WithinSS[i] <- sum(kmeans(km_data,
                                         centers=i)$withinss)
plot(1:9, WithinSS, type="b", xlab="Number of Clusters",
     ylab="Within groups sum of squares")

# Step 02- set non heirarchical cluster
km_fit <-kmeans(km_data,centers = 3)

# Step 03 a- cluster details
km_fit

# Step 03 b- get cluster means
km_fit$centers
km_fit$size
km_fit$withinss 

# Step 04- append cluster assignment
my_iris <- data.frame(my_iris, km_fit$cluster)
str(my_iris)

# Step 05 - cross tab of generated cluster vs original cluster
table(my_iris$Species,my_iris$km_fit.cluster )
