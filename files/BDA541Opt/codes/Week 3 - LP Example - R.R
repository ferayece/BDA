library(lpSolve)
#defining parameters
obj.fun <- c(280,450,600,730,820,280,450,600,730,280,450,600,280,450,280)

#defining constraints
constr <- matrix(c(1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,
                   0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,
                   0,0,1,1,1,0,1,1,1,1,1,1,0,0,0,
                   0,0,0,1,1,0,0,1,1,0,1,1,1,1,0,
                   0,0,0,0,1,0,0,0,1,0,0,1,0,1,1), ncol = 15, byrow=TRUE)

#defining constraints' 
constr.dir <- c(">=", ">=",">=",">=",">=") 

#defining constraints' right hand sides
rhs <- c(15000,10000,20000,5000,25000)

#solving model
prod.sol <- lp("min", obj.fun, constr, constr.dir, rhs, compute.sens = TRUE)

#objective function's optimal value
prod.sol$objval

#optimal values 
prod.sol$solution
