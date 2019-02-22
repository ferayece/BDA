#Q1
WeekdaySales <- c(30,42,34,23,45)
#Q2
Weekdays <- c("M","Tu","W","Th","F")
#Q3
WeekendSales <- c(12,15)
#Q4
Weekend <- c("Sa","Su")
#Q5
Sales <- c(WeekdaySales,WeekendSales)
#Q6
Days <- c(Weekdays,Weekend)
#Q7
summary(Sales)
summary(Days)
#Q8
SalesInTl <- (Sales*5)
#Q9
length(Days)
#Q10
DayNum <- 1:7
#Q11
Sales[3]
#Q12
Sales > 30
#Q13
Days[Sales>30]
#Q14
mean(SalesInTl)
#Q15
max(SalesInTl)
#Q16
sd(SalesInTl)
#Q17
Sales[3] <- NA
SalesInTl <- Sales*5
#Q18
mean(SalesInTl)
max(SalesInTl)
sd(SalesInTl)
#Q19
Sales.df <- data.frame(DayNum,Days,Sales,SalesInTl,stringsAsFactors = FALSE)
#Q20
Sales.df[4,2]
#Q21
Sales.df[,2]
#Q22
Sales.df[4,]
#Q23
summary(Sales.df)
#Q24

#Q25
write.csv(Sales.df,row.names = FALSE)
#Q26
t <- function(x,y){
    (sqrt(x^2+y^2))
}
t(3,4)
