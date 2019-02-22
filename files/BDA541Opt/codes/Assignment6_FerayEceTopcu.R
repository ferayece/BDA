library(mc2d)
library(stats)
library(FinancialMath)
library(triangle)

#https://cran.r-project.org/web/packages/mc2d/mc2d.pdf
#https://cran.r-project.org/web/packages/FinancialMath/FinancialMath.pdf
#https://meredithfranklin.github.io/R-Probability-Distributions.html
#http://data-analytics.net/wp-content/uploads/2014/09/MonteCarloR1.html
#https://cran.r-project.org/web/packages/triangle/triangle.pdf
#https://cran.r-project.org/web/packages/mc2d/mc2d.pdf

#Problem 14-9: IPO
#Base:
stock_cnt <- 40000
suc_fail <- c(1,1,1,1,1)  ##bernolli ,prob=0.7
value_increase <- 1.5 #lognormal distribution mean=1.5% sd=0.5
price <- c(10,10.15,10.30,10.46,10.61,10.77)  ##discrete with prob
final_value <- 10.77*40000
print(final_value)

#sim:
stock_cnt <- 40000
suc_fail <- rbern(1,0.7) # 1 random 
#print(suc_fail)
value_increase <- rlnorm(1,0.015,0.005) #1 random
#print(value_increase)
#print(suc_fail*value_increase)
#price <-  mcstoc(rempiricalD,nsv=5, values=c(10,11,12,13,14,15), prob=c(0.1,0.2,0.3,0.2,0.1,0.1))  
price <- rempiricalD(1, values=c(10,11,12,13,14,15),prob=c(0.1,0.2,0.3,0.2,0.1,0.1)) # 1 random
#print(price)
final_value2 <- suc_fail*value_increase*price
print(stock_cnt * final_value2)
running=suc_fail
print(running)

#Problem 14-12: NPV
##Base 2 14-12:
cost_perc <- 0.65
tax_rate <- 0.36
disc_rate <- 0.1
investment <- c(30000,30000,30000,30000,30000)
revenues <- c(100000,100000,100000,100000,100000)
cost <- revenues*cost_perc
profit <- revenues-cost
taxable_income <- profit-investment
tax <-taxable_income*tax_rate
cash_flow <- profit-investment-tax
NPV(cf0=0,times=c(1,2,3,4,5),cf=cash_flow,i=disc_rate,plot = TRUE)

#Sim:
#cost_perc <- 0.65
#tax_rate <- 0.36
disc_rate <- 0.1
investment <- c(30000,30000,30000,30000,30000)  
revenues <-  rtriangle(n=5,a=60000,b=125000,c=100000)          #triangular dist.
cost_perc <-runif(n=5,min = 0.55,max=0.75) #uniform
cost <- revenues*cost_perc
profit <- revenues-cost
taxable_income <- profit-investment
tax_rate<-rempiricalD(5, values=c(0.36,0.40),prob=c(0.4,0.6))  # custom discrete dist.  rempiricalD(n, values, prob=NULL)
tax <-taxable_income*tax_rate
cash_flow <- profit-investment-tax
NPV(cf0=0,times=c(1,2,3,4,5),cf=cash_flow,i=disc_rate,plot = TRUE)

print("years have positive cashflow:")
for (i in 1:length(cash_flow))
{ 
  if(cash_flow[i]>0)
  {
    print(sprintf("year: %d has positive cash_flow: %f",i,cash_flow[i]))
  }
}
