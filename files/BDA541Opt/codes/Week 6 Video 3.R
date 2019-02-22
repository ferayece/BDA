#install.packages("triangle")
library(triangle) 

rMydist <- function(n) {
  sample(x = c(0.36,0.40), size = n, 
         prob = c(.4,.6), replace=T)
}

investment <- 30000    #150000 over 5 years
NPV = NULL   
Positive_Cash_Flow = NULL
for(i in 1:10000) { 
  yearly_cash_flow = NULL
  yearly_positive = NULL
  for(j in 1:5){
    tax_rate <- rMydist(1)
    annual_revenues <- rtriangle(n=1 ,a = 60000, b = 125000, c = 100000)  
    operating_cost <- runif(n = 1, min = 0.55, max = 0.75)
    yearly_cash_flow[j] = ((annual_revenues * (1 - operating_cost)) - investment) * (1-tax_rate) * ((10/11)^(j-1))
    yearly_positive[j] = yearly_cash_flow[j] > 0
    
  }
  Positive_Cash_Flow[i] = (sum(yearly_positive)) >= 5
  NPV[i] = sum(yearly_cash_flow)
}

hist(NPV,col="wheat")
summary(NPV)
mean(Positive_Cash_Flow)
round(100*mean(NPV<0),2)