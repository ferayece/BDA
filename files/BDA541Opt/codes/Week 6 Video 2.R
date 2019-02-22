install.packages("triangle")
library(triangle) 

rMydist <- function(n) {
  sample(x = c(10,11,12,13,14,15), size = n, 
         prob = c(.1,.2,.3,.2,.1,.1), replace=T)
}

rMydist_2 <- function(n) {
  sample(x = c(1,0), size = n, 
         prob = c(.7,0.3), replace=T)
}
Running5=NULL
Final_Value=NULL   ##This instantiates an object that we will use to store our answers
for(i in 1:100000) {  ##This is how you conduct a loop 1000 time in R
  price <- rMydist(1)
  Running5[i] = 1
  #price <- 10
  shares <- 40000
  stock_increase <- rlnorm(5,0.015,0.005)
  
  for(j in 1:5){
    price <- price * stock_increase[j]
    Running5[i] <- Running5[i] * rMydist_2(1)
  }
  Final_Value[i]<-price * shares   
}

hist(Final_Value,col="wheat")
summary(Final_Value)

hist(Running5,col="wheat")
summary(Running5)
