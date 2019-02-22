#P11-1: Problem about 8 projects in Texas Electronic Company
library(lpSolve)

#a)What is the maximum profit, which project should be selected?
obj.function <- c(36, 82, 29, 16, 56, 61, 48, 41)
constraints <- matrix(c(60, 110, 53, 47, 92, 85, 73, 65, 7, 9, 8, 4, 7, 6, 8, 5), ncol = 8, byrow=TRUE)
constraints.dir <- c("<=", "<=") 
rhs <- c(300, 40)
sol <- lp("max", obj.function, constraints, constraints.dir, rhs, compute.sens = TRUE,all.bin = TRUE)
print(sol$solution)
print(sol)


#b)if P2 and P5  are exclusively selected:
obj.function <- c(36, 82, 29, 16, 56, 61, 48, 41)
constraints  <- matrix(c(60, 110, 53, 47, 92, 85, 73, 65,
                   7, 9, 8, 4, 7, 6, 8, 5,
                   0, 1, 0, 0, 1, 0, 0, 0), ncol = 8, byrow=TRUE)
constraints.dir <- c("<=", "<=", "<=") 
rhs <- c(300, 40, 1)
sol <- lp("max", obj.function, constraints, constraints.dir, rhs, compute.sens = TRUE,all.bin = TRUE)
print(sol$solution)
print(sol)

#c)if at least two of p5,p6,p7,p8 are selected:
obj.function <- c(36, 82, 29, 16, 56, 61, 48, 41)
constraints <- matrix(c(60, 110, 53, 47, 92, 85, 73, 65,
                   7, 9, 8, 4, 7, 6, 8, 5,
                   0, 1, 0, 0, 1, 0, 0, 0,
                   0, 0, 0, 0, 1, 1, 1, 1), ncol = 8, byrow=TRUE)
constraints.dir <- c("<=", "<=", "<=", ">=") 
rhs <- c(300, 40, 1, 2)
#solving model
sol <- lp("max", obj.function, constraints, constraints.dir, rhs, compute.sens = TRUE,all.bin = TRUE)
print(sol$solution)
print(sol)
