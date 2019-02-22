library(lpSolve)
###########
obj.fun <- c(8, 6, 10)
constr <- matrix(c(0.2, 0.4, 0.6, 0.6, 0.2, 0.5, 0.4, 0.4, 0.4), ncol = 3, byrow=TRUE)
constr.dir <- c("<=", "<=", "<=") 
rhs <- c(150, 150, 600)
#solving model
prod.sol <- lp("max", obj.fun, constr, constr.dir, rhs, compute.sens = TRUE)

print(prod.sol$solution)
## [1] 150 300   0
print(prod.sol)
## Success: the objective function is 3000

############################Example: P9-7 #########################
revenue <- c(29.50,28.00,29.50,28)
cost <- c(18.80,16.00,21.50,21.50)
obj.function <- revenue-cost
#print(obj.function)
constraints <- matrix(c(0.15,0.1,0,0,0.2,0.2,0,0,0.1,0.15,0,0,1,0,1,0,0,1,0,1,0,0,1,1),ncol=4,byrow = TRUE)
#constraints <- matrix(c(0.15,0.1,0,0,0.2,0.2,0,0,0.1,0.15,0,0,1,0,1,0,0,1,0,1,0,0,1,1),nrow=6,byrow = TRUE)
#print(constraint)
constraints.dir <- c("<=","<=","<=","<=","<=")
#print(constraints.dir)
rhs <- c(2000,4200,2500,20000,10000,20000)
prod.sol<- lp("max", obj.function, constraints, constraints.dir, rhs, compute.sens = TRUE)
print(prod.sol$solution)
print(prod.sol)

############################Example: P9-7 with Multiple Parameter Value -change Fabrication #########################

revenue <- c(29.50,28.00,29.50,28)
cost <- c(18.80,16.00,21.50,21.50)
obj.function <- revenue-cost
#print(obj.function)
constraints <- matrix(c(0.15,0.1,0,0,0.2,0.2,0,0,0.1,0.15,0,0,1,0,1,0,0,1,0,1,0,0,1,1),ncol=4,byrow = TRUE)
#constraints <- matrix(c(0.15,0.1,0,0,0.2,0.2,0,0,0.1,0.15,0,0,1,0,1,0,0,1,0,1,0,0,1,1),nrow=6,byrow = TRUE)
#print(constraint)
constraints.dir <- c("<=","<=","<=","<=","<=")
#print(constraints.dir)
fabrication<-2000
while(fabrication<=2500){
  rhs <- c(fabrication,4200,2500,20000,10000,20000)
  prod.sol<- lp("max", obj.function, constraints, constraints.dir, rhs, compute.sens = TRUE)
  print(prod.sol$solution)
  print(prod.sol)
  #print(prod.sol$objval)
  #cat("Current result of obj  function:",prod.sol$objval)
  fabrication <- fabrication +100
}

