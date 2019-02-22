#install.packages("triangle")
library(triangle) 

rMydist <- function(n) {
  sample(x = c(0.05,0.12,0.15), size = n, 
         prob = c(.2,.5,.3), replace=T)
}

TotalProfit <- NULL
FixedCost <- 100000

TotalRows <- 40
FirstClassRows <- 3
TouristRows <- TotalRows - 2*FirstClassRows
FirstClassSeats <- FirstClassRows * 4
TouristSeats <- TouristRows * 6

PriceFirst <- NULL
PriceFirst[1] <- 400 #Boston-Atlanta
PriceFirst[2] <- 400 #Atlanta-Chicago
PriceFirst[3] <- 450 #Chicago-Boston
PriceTourist <- NULL
PriceTourist[1] <- 175 #Boston-Atlanta
PriceTourist[2] <- 150 #Atlanta-Chicago
PriceTourist[3] <- 200 #Chicago-Boston

MinDemand <- NULL
MinDemand[1] <- 160 #Boston-Atlanta
MinDemand[2] <- 140 #Atlanta-Chicago
MinDemand[3] <- 150 #Chicago-Boston
MLDemand <- NULL
MLDemand[1] <- 180 #Boston-Atlanta
MLDemand[2] <- 200 #Atlanta-Chicago
MLDemand[3] <- 200 #Chicago-Boston
MaxDemand <- NULL
MaxDemand[1] <- 220 #Boston-Atlanta
MaxDemand[2] <- 240 #Atlanta-Chicago
MaxDemand[3] <- 225 #Chicago-Boston

#Initialize
FirstClassFraction <- NULL
TotalDemand <- NULL
FirstClassDemand <- NULL
TouristDemand <- NULL
FirstClassSales <- NULL
TouristSales <- NULL
FirstClassRev <- NULL
TouristRev <- NULL
Rev <- NULL

TotalProfit <- NULL

for(i in 1:10000) { 
  
  TotalRev = 0
  
  for(j in 1:3){
    FirstClassFraction[j] <- rMydist(1)
    TotalDemand[j] <- rtriangle(n=1 ,a = MinDemand[j], b = MaxDemand[j], c = MLDemand[j])  
    FirstClassDemand[j] <- floor(TotalDemand[j]*FirstClassFraction[j])
    TouristDemand[j] <- TotalDemand[j] - FirstClassDemand[j]
    FirstClassSales[j] <- min(FirstClassDemand[j],FirstClassSeats)
    TouristSales[j] <- min(TouristDemand[j],TouristSeats)
    FirstClassRev[j] = FirstClassSales[j] * PriceFirst[j]
    TouristRev[j] = TouristSales[j] * PriceTourist[j]
    Rev[j] = FirstClassRev[j] + TouristRev[j]
    TotalRev = TotalRev + Rev[j]
  
    }
  
  TotalProfit[i] = TotalRev - FixedCost
  
}
hist(TotalProfit,col="wheat")
mean(TotalProfit)
