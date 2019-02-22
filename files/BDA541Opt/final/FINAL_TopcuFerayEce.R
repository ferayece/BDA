library(lpSolve)
library(triangle) 
###########   Final-Q1: Linear Programming####################
##Problem 1-A:  Fancy Skateboard Manufacturing 

obj.fun <- c(80,30)
constr <- matrix(c(1,1,
                   1,-1,
                   2,1,
                   3,4), nrow = 4, byrow=TRUE)
constr.dir <- c(">=", "<=", "<=", "<=") 
rhs <- c(600, 300, 1200,2800)
#solving model
prod.sol <- lp("max", obj.fun, constr, constr.dir, rhs, compute.sens = TRUE)

print(prod.sol$solution)

print(prod.sol)

###########   Final-Q3: Simulation ####################
##Problem 3-A:  Bookstore Prod.

#Part 1: just calculate the profit.

fixed_cost <- 72000
stock <- 1500
planned_prod <-1000
total_book <- stock+planned_prod

unit_cost <- 10
unit_price <- 30

demand <- 2000 # made up value min=1250,max=5000,likely=2500

satisfied_demand <- min(demand,total_book)
penalty_cost_total <- 2*(max(0,demand-satisfied_demand)) #cost
leftover_profit_total <- 3*(max(0,total_book-satisfied_demand)) # revenue

total_revenue <- satisfied_demand * unit_price + leftover_profit_total
total_cost <- fixed_cost + planned_prod * unit_cost + penalty_cost_total

total_profit <- total_revenue - total_cost

print(total_profit)


#Part 2: Simulate the total profit for bookstore:
fixed_cost <- 72000
stock <- 1500
planned_prod <-1000
total_book <- stock+planned_prod

unit_cost <- 10
unit_price <- 30

profit_sim=NULL

for(i in 1:100000) { 
   demand <- round(rtriangle(n=1 ,a = 1250, b = 5000, c = 2500)) #min,max,mostlikely
   #demand <- rtriangle(n=1 ,a = 1250, b = 5000, c = 2500)
   satisfied_demand <- min(demand,total_book)
   
   penalty_cost_total <- 2*(max(0,demand-satisfied_demand)) #cost
   leftover_profit_total <- 3*(max(0,total_book-satisfied_demand)) # revenue
   
   total_revenue <- satisfied_demand * unit_price + leftover_profit_total
   total_cost <- fixed_cost + planned_prod * unit_cost + penalty_cost_total
   
   total_profit <- total_revenue - total_cost
   
   profit_sim[i] <- total_profit
  
  }

hist(profit_sim,col="wheat")
summary(profit_sim)
