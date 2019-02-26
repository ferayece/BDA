# Chapter 9 Lab: Support Vector Machines

# Support Vector Classifier
#Create the data set
set.seed(1)
x=matrix(rnorm(20*2), ncol=2)
y=c(rep(-1,10), rep(1,10))
x[y==1,]=x[y==1,] + 1
#plot the data set
plot(x, col=(3-y))
#Encode the response variable as factor  GEREKLI
dat=data.frame(x=x, y=as.factor(y))
#load the library
library(e1071)
#estimate the svm

#?svm #özellikleri gör.
svmfit=svm(y~., data=dat, 
           kernel="linear",   #en basiti.
           cost=10,
           scale=FALSE)  #TRUE olursa tum predictorleri scale eder. 
plot(svmfit, dat)   
#turkuazda yuvarlak kalanlar support degil;çizginin çizilmesine etkisi yok bunlarin.
#marjine degenler supporttur!
#en altta turkuazda kirmizi bir x var; missclasify edilen ama support olan bir tane!

#support vectors
svmfit$index
#show summary of SVM
summary(svmfit)
#we change cost to smaller value
# misscalification'a izin verdik cost'u düsürerek!
svmfit=svm(y~., data=dat, kernel="linear", cost=0.1,scale=FALSE)
plot(svmfit, dat)
#marjini büyüttük. çok daha fazla supportumuz var.

svmfit$index

######################## grid searchlü:
##e1071 paketinin içinde gridsearch var 10-fold CV ile grid search yapiyor.
#simdi hanig c daha iyi diye çalistiralim?

set.seed(1)


tune.out=tune(svm,y~.,data=dat,
              kernel="linear",
              ranges=list(cost=c(0.001, 0.01, 0.1, 1,5,10,100)))
summary(tune.out)  ##erro u en düsük ya da accuracy si en yüksek olani al.
bestmod=tune.out$best.model #en iyi modeli aldik.
summary(bestmod)   #cost=0.001 olan.

#form test set
xtest=matrix(rnorm(20*2), ncol=2)
ytest=sample(c(-1,1), 20, rep=TRUE)
xtest[ytest==1,]=xtest[ytest==1,] + 1
testdat=data.frame(x=xtest, y=as.factor(ytest))

ypred=predict(bestmod,testdat)
table(predict=ypred, truth=testdat$y)  #sadece 1 yanlis tahmin.

#marjini açalim costu düsürüp:
#change cost to 0.01
svmfit=svm(y~., data=dat, kernel="linear", cost=.01,scale=FALSE)
ypred=predict(svmfit,testdat)
table(predict=ypred, truth=testdat$y)  # 2 tane yanlis bildi, hata payimiz artti.


###########kerneli degistirelim:
##niye kernel kullaniyoruz?  --> non-linear hyperplane gerekince kerneli degistir. 
#radial --> en popi. sigmoid
#tüm kerneler için gereken parametreler için ?svm yap.
# Support Vector Machine
set.seed(1)
x=matrix(rnorm(200*2), ncol=2)
x[1:100,]=x[1:100,]+2
x[101:150,]=x[101:150,]-2
y=c(rep(1,150),rep(2,50))
dat=data.frame(x=x,y=as.factor(y))

plot(x, col=y)  ## radial oldugu belli bu datasetin...kirmizilar halka gibi içerde.

train=sample(200,100)  #200ün 100ünü seç.

svmfit=svm(y~., data=dat[train,], 
           kernel="radial",
           gamma=1,
           cost=1)
plot(svmfit, dat[train,])

summary(svmfit)

#marjini büyüt=costu küçült:

svmfit=svm(y~., data=dat[train,], 
           kernel="radial",
           gamma=1,
           cost=0.1)
plot(svmfit, dat[train,])   ## hatayi görebilirsin.


#marjini küçült=costu büyüt:

svmfit=svm(y~., data=dat[train,], 
           kernel="radial",
           gamma=1,
           cost=10)
plot(svmfit, dat[train,])   ## 1 den çok farkli olmadi. 

svmfit=svm(y~., data=dat[train,], 
           kernel="radial",gamma=1,cost=1e5)
plot(svmfit,dat[train,])


#gammayi küçült:

svmfit=svm(y~., data=dat[train,], 
           kernel="radial",
           gamma=0.1,
           cost=1)
plot(svmfit, dat[train,])   ## aaaiiy marjini kaybettik resmen rezalet.


#gammayi küçült:

svmfit=svm(y~., data=dat[train,], 
           kernel="radial",
           gamma=10,
           cost=1)
plot(svmfit, dat[train,])   ## gammayi yükselttikçe marjinal bi elips çikti. demek ki gamma 1-10 arasi bi deger olmali. 

## deneyelim:

svmfit=svm(y~., data=dat[train,], 
           kernel="radial",
           gamma=1,
           cost=1)
plot(svmfit, dat[train,])

summary(svmfit)   # kaç tane svector oldugunu gorebilirsin.



## gridsearch for tuning of gamma and c: 
set.seed(1)
tune.out=tune(svm, y~., data=dat[train,], kernel="radial", ranges=list(cost=c(0.1,1,10,100,1000),gamma=c(0.5,1,2,3,4)))
summary(tune.out) #cost:1 gamma:2 en iyi sonuc. 

#en iyi modelle test yapalim:
table(true=dat[-train,"y"], pred=predict(tune.out$best.model,newx=dat[-train,])) #39 missclasiffied. (0.39) --> (21+18)/(56+21+18+5)
plot(tune.out$best.model, dat[train,])


##polynomial:
#degree,c,gamma

#küçük degree ise yaramiyore.
svmfit=svm(y~., data=dat[train,], 
           kernel="polynomial",
           gamma=1,
           cost=1,
           degree=10)
plot(svmfit, dat[train,])

#tune for polynomial:
#zorlu bir yol.
set.seed(1)
tune.out=tune(svm, y~., data=dat[train,], kernel="polynomial", ranges=list(cost=c(0.1,1,10),gamma=c(0.5,1),degree=c(5,10)))
summary(tune.out) #1,1,10 en iyi verdi ama sikinti.

#en iyi modelle test yapalim:
table(true=dat[-train,"y"], pred=predict(tune.out$best.model,newx=dat[-train,])) #39 missclasiffied. (0.39) --> (21+18)/(56+21+18+5)
plot(tune.out$best.model, dat[train,])


## 3 kategorimiz olsaydi classify etmeye çalistigimiz. Arka arkaya svm yapmamiz lazim.
# e1071 pakedi; 3 kategori verince 3'ün 2'li kombinasyonu kadar svm yapar.
# 3 svm'in ortalamasini aliyor gibi düsün. bir observation'a her svm'de söylenen class i aliyor en çok ne olarak isaretlendiyse onun classi o oluyor.

# ROC Curves
#ROC, accuracysini gösterir.
#line ne kadar kösede olursa o akdar iyi, ne kadar 45 dereceye yakin olursa line o kadar kötü.
library(ROCR)
rocplot=function(pred, truth, ...){
  predob = prediction(pred, truth)
  perf = performance(predob, "tpr", "fpr")
  plot(perf,...)}

svmfit.opt=svm(y~., data=dat[train,], kernel="radial",gamma=2, cost=1,decision.values=T)
fitted=attributes(predict(svmfit.opt,dat[train,],decision.values=TRUE))$decision.values

par(mfrow=c(1,2))
rocplot(fitted,dat[train,"y"],main="Training Data")

svmfit.flex=svm(y~., data=dat[train,], kernel="radial",gamma=50, cost=1, decision.values=T)
fitted=attributes(predict(svmfit.flex,dat[train,],decision.values=T))$decision.values
rocplot(fitted,dat[train,"y"],add=T,col="red")

fitted=attributes(predict(svmfit.opt,dat[-train,],decision.values=T))$decision.values
rocplot(fitted,dat[-train,"y"],main="Test Data")

fitted=attributes(predict(svmfit.flex,dat[-train,],decision.values=T))$decision.values
rocplot(fitted,dat[-train,"y"],add=T,col="red")


#######
# SVM with Multiple Classes   3 kategorili. 

set.seed(1)
x=rbind(x, matrix(rnorm(50*2), ncol=2))
y=c(y, rep(0,50))
x[y==0,2]=x[y==0,2]+2

dat=data.frame(x=x, y=as.factor(y))

par(mfrow=c(1,1))
plot(x,col=(y+1))

svmfit=svm(y~., data=dat, kernel="radial", cost=10, gamma=1)
plot(svmfit, dat)

# Application to Gene Expression Data

library(ISLR)
names(Khan)
dim(Khan$xtrain)
dim(Khan$xtest)
length(Khan$ytrain)
length(Khan$ytest)

dat=data.frame(x=Khan$xtrain, y=as.factor(Khan$ytrain))
out=svm(y~., data=dat, kernel="linear",cost=10)
summary(out)

table(out$fitted, dat$y)

dat.te=data.frame(x=Khan$xtest, y=as.factor(Khan$ytest))
pred.te=predict(out, newdata=dat.te)
table(pred.te, dat.te$y)

# In this problem, you will use support vector approaches in order to predict whether a given car gets high or low gas
#mileage based on the “Auto” data set.

#Data preparation:

library(ISLR)
library(e1071) #gridsearch var.
library(caret) #gridsearch var.

var <- ifelse(Auto$mpg>median(Auto$mpg),1,0)
Auto$mpgLevel <- as.factor(var)

#str(Auto)

#a.Create a binary variable that takes on a 1 for cars with gas mileage above the median, and a 0 for cars with gas mileage below the median.


var <- ifelse(Auto$mpg>median(Auto$mpg),1,0)
Auto$mpgLevel <- as.factor(var)
str(Auto)


#b.Fit a support vector classifier to the data with various values of “cost”, in order to predict whether a car gets high of low gas mileage.
#Report the cross-validation errors associated with different values of this parameter. Comment on your results.

#split as train test:
train <- sample(392,200)
AutoTrain <- Auto[train,c(-1,-9)] #mpg'yi çikar. name'i çikar anlamsiz.
AutoTest <- Auto[-train,]

#train svm:
set.seed(1)
svmTrain <- svm(mpgLevel~., data=AutoTrain, kernel="linear", cost=.01,scale=FALSE)

#predict:
ypred=predict(svmTrain,AutoTest)
table(predict=ypred, truth=AutoTest$mpgLevel)
#Accuracy: 1-(4/192) hiç bi kolonu drop etmedigimizde.

#tune:
set.seed(1)
tune.out=tune(svm, mpgLevel~., data=AutoTrain, kernel="linear", ranges=list(cost=c(0.1,1,10,100,1000),gamma=c(0.5,1,2,3,4)))
summary(tune.out)

#summary of best tuned model:
bestmod=tune.out$best.model #en iyi modeli aldik.
summary(bestmod)

#plot edemedik çünkü 2 den fazla boyut var.


#predict:
ypred=predict(tune.out$best.model,AutoTest)
table(predict=ypred, truth=AutoTest$mpgLevel)

#Accuracy:1-(2/192) hiç bi kolonu drop etmedigimizde.
#Accuracy:1-((14+4)/192)

#c.Now repeat (b), this time using SVMs with radial and polynomial basis kernels, with different values of “gamma” and “degree” and “cost”. 
#Comment on your results.

#Radial: 
set.seed(1)
tune.out=tune(svm, mpgLevel~., data=AutoTrain, kernel="radial", ranges=list(cost=c(0.1,1,10,100,1000),gamma=c(0.5,1,2,3,4)))
summary(tune.out)

#summary of best tuned model:
bestmod=tune.out$best.model #en iyi modeli aldik.
summary(bestmod)

#plot edemedik çünkü 2 den fazla boyut var.


#predict:
ypred=predict(tune.out$best.model,AutoTest)
table(predict=ypred, truth=AutoTest$mpgLevel)


#Radial accuracy: 1- ((6+7)/(6+7+92+87))   0.9322

#Polynomial: 
set.seed(1)
tune.out=tune(svm, mpgLevel~., data=AutoTrain, kernel="polynomial", ranges=list(cost=c(1,10),gamma=c(0.5,1,10),degree=c(0.1,5)))
summary(tune.out)

#summary of best tuned model:
bestmod=tune.out$best.model #en iyi modeli aldik.
summary(bestmod)

#plot edemedik çünkü 2 den fazla boyut var.


#predict:
ypred=predict(tune.out$best.model,AutoTest)
table(predict=ypred, truth=AutoTest$mpgLevel)


#Polynomial accuracy: 1- ((3+19)/192)   0.8854

##caret ile : 
#https://topepo.github.io/caret/model-training-and-tuning.html


bootControl <- trainControl(number=20)  # 20 bootstrap, 10 yazsam 10 fold CV.
set.seed(2)
svmFit <- train(AutoTrain[,-8],AutoTrain[,8],
                method="svmRadial",tuneLength=5,
                trControl=bootControl,scaled=TRUE)
svmFit

#This problem involves the “OJ” data set which is part of the ISLR package.
#a.Create a training set containing a random sample of 800 observations, and a test set containing the remaining observations.
#b.Fit a support vector classifier to the training data using “cost” = 0.01, with “Purchase” as the response and the other variables as predictors. Use the summary() function to produce summary statistics, and describe the results obtained.
#c.What are the training and test error rates ?
#d.Use the tune() function to select an optimal “cost”. Consider values in the range 0.01 to 10.
#e.Compute the training and test error rates using this new value for “cost”.
#f.Repeat parts (b) through (e) using a support vector machine with a radial kernel. Use the default value for “gamma”.
#g.Repeat parts (b) through (e) using a support vector machine with a polynomial kernel. Set “degree” = 2.
#h.Overall, which approach seems to give the best results on this data ?