###### Part 1: Questions about Statistical and Optimization Techniques for Laundry Portfolio Optimization at Procter & Gamble ##########
# 1)) As I understand, they decide the mixture ingredient compositions at the end of the  predictive models. It means P&G produce the some products 
# according to output of this solution, so why did not mention about health? 
# Is not health an important constraint while the output of the modeling is the mixture of some chemical ingredients?
# 2)) Under the 'Implementation and Usage' header, There is 'Team and Workflow' part. On this part, there are 9 step for implementation. 
#8.step is 'Churn and Analysis'. As ı understand on this step, all team members are enforcing the optimization engine with some scenerios.
#I did not understand why all team members should strain the engine, why not just functional experts?

#########********************************************#####################################

#Part2: I r-ify the example in video 3. (PRbolem 9-7 in Problem Sets directory.) 
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

############################Example: P9-7 with Multiple Parameter Value -change value of Fabrication #########################

revenue <- c(29.50,28.00,29.50,28)
cost <- c(18.80,16.00,21.50,21.50)
obj.function <- revenue-cost
constraints <- matrix(c(0.15,0.1,0,0,0.2,0.2,0,0,0.1,0.15,0,0,1,0,1,0,0,1,0,1,0,0,1,1),ncol=4,byrow = TRUE)
constraints.dir <- c("<=","<=","<=","<=","<=")
fabrication<-2000
while(fabrication<=2500){
  rhs <- c(fabrication,4200,2500,20000,10000,20000)
  prod.sol<- lp("max", obj.function, constraints, constraints.dir, rhs, compute.sens = TRUE)
  print(prod.sol$solution)
  print(prod.sol)
  #cat("Current result of obj  function:",prod.sol$objval)
  fabrication <- fabrication + 50
}