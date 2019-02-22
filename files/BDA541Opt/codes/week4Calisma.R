#linear programming:
library(lpSolve)
obj_coefficient <- c(10,15,16)
constraints <-matrix(c(1,1,0,5,10,12,10,6,2),ncol =3,byrow=TRUE)
constraint.dir <- c("<=","<=","<=")
rhs <- c(20,180,200)
sol <- lp("max", obj_coefficient, constraints, constraint.dir,rhs,all.int = TRUE, compute.sens = TRUE)
print(sol$solution)
print(sol)


library(ompr)
library(magrittr)
library(ROI)
library(ROI.plugin.symphony)
model <- MIPModel() %>% 
          add_variable(c(x1,x2,x3),type="integer",lb=0) %>% 
          add_variable(c(y1,y2,y3),type="binary",lb=0) %>% 
          add_constraint(x1 + x2 <=20) %>%
          add_constraint(5*x1+10*x2+12*x3<=180) %>%
          add_constraint(x1<=y1)  %>%
  solve_model(with_ROI(solver = "symphony",verbosity=1))


result <- solve_model(model,solver=)