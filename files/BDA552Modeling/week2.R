# The Validation Set Approach

library(ISLR)
set.seed(1)  # CV yapmadan seedi mutlaka set et. herkes aynı sonucu alsın diye!
train=sample(392,196)  #sampling 392 içinden 196 lık bir sample seçiyoruz. (196 lık vektor)
lm.fit=lm(mpg~horsepower,data=Auto,subset=train)
summary(lm.fit)
attach(Auto) # env variable yaptık.
mean((mpg-predict(lm.fit,Auto))[-train]^2) #MSI hesaplıyoruz.RMSI(root) da kullanılabilir. (mean((fit- predicted)^2):mean square error)
lm.fit2=lm(mpg~poly(horsepower,2),data=Auto,subset=train) # bu ilişkinin lineer olmadığını biliyoruz. karesi işimize yarar mı ?
mean((mpg-predict(lm.fit2,Auto))[-train]^2) # Error düştü. Prediction da improvement var. Güzel!
lm.fit3=lm(mpg~poly(horsepower,3),data=Auto,subset=train)
mean((mpg-predict(lm.fit3,Auto))[-train]^2) # Error çok küçük düştü, gerek yok. 
#Bu yöntemde sorun nedir? >> çıkan sonuçlar sample ı bölmemize çok duyarlı. 
#Sample() metodu ile böldük çünkü. Buna validation denir. Bize cross validation lazım.

#Başka sample set seçince başka sonuç çıkaracak. Çünkü düz validation yapıyoruz.
set.seed(2)
train=sample(392,196)
lm.fit=lm(mpg~horsepower,subset=train)
mean((mpg-predict(lm.fit,Auto))[-train]^2)
lm.fit2=lm(mpg~poly(horsepower,2),data=Auto,subset=train)
mean((mpg-predict(lm.fit2,Auto))[-train]^2)
lm.fit3=lm(mpg~poly(horsepower,3),data=Auto,subset=train)
mean((mpg-predict(lm.fit3,Auto))[-train]^2) # küp daha iyi çıktı şimdi de....

##Cross validation:

# amacımız datayı split etmek değil; amacımız modelleri karşılaştırmak! 
# en iyi modeli bulmak amacımız !!!
# Varyansı düşük modeli bulalım diye yapıyoruz!
# Leave-One-Out Cross-Validation

glm.fit=glm(mpg~horsepower,data=Auto) #family=binomial ise logistict reg yapar. Bu halde Lineer Reg yapar.
#GLM içinde default CV olduğu için Lm yerime glm kullanıyoruz!
coef(glm.fit)
lm.fit=lm(mpg~horsepower,data=Auto)
coef(lm.fit)
library(boot) # for CV.
glm.fit=glm(mpg~horsepower,data=Auto)
cv.err=cv.glm(Auto,glm.fit) #otomatik leave one out CV yapıyoruz.
cv.err$delta  #CV error'u.
cv.error=rep(0,5)
for (i in 1:5){  #5*396 kere çalıştı bu regression. 395 i tuttu 1 ini test yaptı. O yüzden. (n e bölüyor leaveoneout)
  glm.fit=glm(mpg~poly(horsepower,i),data=Auto)
  cv.error[i]=cv.glm(Auto,glm.fit)$delta[1]
} 
cv.error # en iyisi 5.si. ama 2.yi tercih etmek daha makul çok fark yok arasında.
#2. derece polynomial yeterli bizim için. 
#yüksek varyanslı -sensitive- model kullanıyorsanız K-FOLD daha avantajlı.

# k-Fold Cross-Validation

set.seed(17)
cv.error.10=rep(0,10)
for (i in 1:10){ #10*10 = 100 kere çalışacak model.
  glm.fit=glm(mpg~poly(horsepower,i),data=Auto)
  cv.error.10[i]=cv.glm(Auto,glm.fit,K=10)$delta[1]
}
cv.error.10
#5. en iyisi. 

#Chapter 5: In-Class Question

# In Chapter 4, we used logistic regression to predict the probability of
# default using income and balance on the Default data set. We will
# now estimate the test error of this logistic regression model using the
# validation set approach. Do not forget to set a random seed before
# beginning your analysis.

#(a) Fit a logistic regression model that uses income and balance to
#predict default.
library(ISLR)
attach(Default)
str(Default)
glm.fit=glm(default~income+balance,data = Default,family=binomial)
summary(glm.fit)



#(b) Using the validation set approach, estimate the test error of this
#model. In order to do this, you must perform the following steps:

  #i. Split the sample set into a training set and a validation set.

set.seed(2)
train=sample(10000,5000)  #or sample(dim(Default)[1],dim(Default)[1]/2)

  
# ii. Fit a multiple logistic regression model using only the training observations.

glm.fit=glm(default~income+balance,data = Default,family=binomial,subset=train)
summary(glm.fit)

#iii. Obtain a prediction of default status for each individual in
#the validation set by computing the posterior probability of
#default for that individual, and classifying the individual to
#the default category if the posterior probability is greater than 0.5.
#posterior probability: prediction yaptıktan sonra verilen olasılık.
probs <- predict(glm.fit,newdata=Default[-train,],type="response") 
# type ı silersek regression yapar, böyle classification.
#newdata=Default[-train,] = test datası kaldı. 
pred.glm <- rep("No",length(probs))  ##5000 No diyorum.
pred.glm[probs>0.5] <-"Yes" #eğer 0.5 den büyükse Yes e çevirir.
pred.glm


#iv. Compute the validation set error, which is the fraction of
#the observations in the validation set that are misclassified.
mean(pred.glm != Default[-train,]$default) # ne kadar hata  yaptığımızı gösterdi.


#(c) Repeat the process in (b) three times, using three different splits
#of the observations into a training set and a validation set. Comment
#on the results obtained.

#aynısını farklı seed de yapın 3 kere diyor.

# (d) Now consider a logistic regression model that predicts the probability
# of default using income, balance, and a dummy variable
# for student. Estimate the test error for this model using the validation
# set approach. Comment on whether or not including a
# dummy variable for student leads to a reduction in the test error
# rate.
# fazladan student ı ekle.

glm.fit=glm(default~income+balance+student,data = Default,family=binomial,subset=train)
summary(glm.fit)
probs <- predict(glm.fit,newdata=Default[-train,],type="response") 
# type ı silersek regression yapar, böyle classification.
#newdata=Default[-train,] = test datası kaldı. 
pred.glm <- rep("No",length(probs))  ##5000 No diyorum.
pred.glm[probs>0.5] <-"Yes" #eğer 0.5 den büyükse Yes e çevirir.
pred.glm
mean(pred.glm != Default[-train,]$default) # hata arttı student koyunca.

#Chapter 6

#lineer dünya için feature selection yöntemleri. 

# Best Subset Selection 
# tüm olasıkları deniyoruz.Disad: yavaş. 20 feature, milyon kere denemek demek.
#o yüzden bunu tercih etmiyoruz. 

library(ISLR)
#fix(Hitters) # baseball vuruşcularının gelirleri. Salary tahminlicez.
attach(Hitters)
names(Hitters) #features
dim(Hitters)
sum(is.na(Hitters$Salary))
Hitters=na.omit(Hitters) # na içeren tüm rowları atar!
dim(Hitters)
sum(is.na(Hitters))
library(leaps)  # selection yöntemleri.
regfit.full=regsubsets(Salary~.,Hitters) # tüm subseti deniyoruz. FULL: 
summary(regfit.full) # 1 değişken seçeceksem hangisi seçilmeli, row a bak yıldızlıyor!
regfit.full=regsubsets(Salary~.,data=Hitters,nvmax=19) # 19 değişkenin en iyisi.
reg.summary=summary(regfit.full)
#reg.summary
names(reg.summary)
reg.summary$rsq
par(mfrow=c(2,2))
plot(reg.summary$rss,xlab="Number of Variables",ylab="RSS",type="l")
plot(reg.summary$adjr2,xlab="Number of Variables",ylab="Adjusted RSq",type="l")
which.max(reg.summary$adjr2)
points(11,reg.summary$adjr2[11], col="red",cex=2,pch=20)
plot(reg.summary$cp,xlab="Number of Variables",ylab="Cp",type='l')
which.min(reg.summary$cp)
points(10,reg.summary$cp[10],col="red",cex=2,pch=20)
which.min(reg.summary$bic)
plot(reg.summary$bic,xlab="Number of Variables",ylab="BIC",type='l')
points(6,reg.summary$bic[6],col="red",cex=2,pch=20)
plot(regfit.full,scale="r2")
plot(regfit.full,scale="adjr2")
plot(regfit.full,scale="Cp")
plot(regfit.full,scale="bic")
coef(regfit.full,6)

# Forward and Backward Stepwise Selection
#trainde en iyisni full veriyor çünkü tüm kombinasyonları deniyor!
#test de overfitting yapabilir; en iyisni seçmeyebilir full. (CV de de durum aynı.)
#sürekli bir feature arttırıyor. Full'den dezavantajı; bir değpişkeni alınca fw onu çıkarmıyor.
#farklı combination'ları denemediği için fw en iyisini veremez.
#CRBI aldı, bir kombinasyonda bu feature kötüleşti ama bunu asla çıkarmaz!!!
regfit.fwd=regsubsets(Salary~.,data=Hitters,nvmax=19,method="forward") 
summary(regfit.fwd)
#backwardın sorunu da bir feature ı atınca, bi daha almıyor. Değişik komb kaçırır.
regfit.bwd=regsubsets(Salary~.,data=Hitters,nvmax=19,method="backward")
summary(regfit.bwd)

coef(regfit.full,7) # the best 7 for full.
coef(regfit.fwd,7) # the best 7 for fwd.
coef(regfit.bwd,7) # the best 7 for bwd.

#Choosing Among Models

set.seed(1)
train=sample(c(TRUE,FALSE), nrow(Hitters),rep=TRUE) #nrow kadar true, false ata.
test=(!train)

regfit.best=regsubsets(Salary~.,data=Hitters[train,],nvmax=19)
test.mat=model.matrix(Salary~.,data=Hitters[test,]) #test dataseti alıp matrix halinde test.mat atıyor.
#matrix e neden çeviriyoruz? predict'e input olarak verebilmek için.

####manuel validation=coef*feature + ...  
val.errors=rep(NA,19)
#hangi modelin prediction error ı en düşük, manuel hesaplıyoruz.
for(i in 1:19){
  coefi=coef(regfit.best,id=i)  #regfit.best;19 modelin tüm coef lerini tutar. regfit.best[1] ilk modelin coef leri.
  pred=test.mat[,names(coefi)]%*%coefi
  val.errors[i]=mean((Hitters$Salary[test]-pred)^2)
}
val.errors
which.min(val.errors) # hangisi en küçük error ı verdi?
coef(regfit.best,10) #coef of 10 featureluk model.

##manuel prediction=.. (pakette olmadığı için elle yazıyoruz.)

predict.regsubsets=function(object,newdata,id,...){  ## env a bir function tanımladık.
  form=as.formula(object$call[[2]])
  mat=model.matrix(form,newdata)
  coefi=coef(object,id=id)
  xvars=names(coefi)
  mat[,xvars]%*%coefi
}

regfit.best=regsubsets(Salary~.,data=Hitters,nvmax=19) ## full subset again.
coef(regfit.best,10)  # coef of 10th model. 10 featureluk model.

#Apply k-fold CV:
k=10
set.seed(1)
folds=sample(1:k,nrow(Hitters),replace=TRUE) #replace=TRUE: torbaya geri atma. FALSE dersen tüm dataseti kullanır.

cv.errors=matrix(NA,k,19, dimnames=list(NULL, paste(1:19))) # 10*19 luk matrix. Her row 1 CV. 1. feature için 10 kere CV. MSE'nin avg'sini bulcaz.

for(j in 1:k){
  best.fit=regsubsets(Salary~.,data=Hitters[folds!=j,],nvmax=19)
  for(i in 1:19){
    pred=predict(best.fit,Hitters[folds==j,],id=i)
    cv.errors[j,i]=mean( (Hitters$Salary[folds==j]-pred)^2)
  }
}
mean.cv.errors=apply(cv.errors,2,mean)
mean.cv.errors
which.min(mean.cv.errors)   # CV'de 11 en iyi çıktı ama 10 ile çok fark yok.

par(mfrow=c(1,1))
plot(mean.cv.errors,type='b')
reg.best=regsubsets(Salary~.,data=Hitters, nvmax=19)
coef(reg.best,11)


# Chapter 6 Lab 2: Ridge Regression and the Lasso

#Regularization: Linear equation daki regularization'a karşılık gelir.
#Featurelar önüne lambda koyuyoruz; (shrikange penalty); lambda arttıkça o feature'un modele seçilme olasılığı düşüyor.
#Ridge: 0'a düşürmüyor.
#Lasso: 0'a indiriyor. Açıklanabilirlik açısından lasso daha iyi. Kesinlikle şunlar işe yaramaz diyebiliyorsunuz.
#Lasso sıfırladığı için Ridge biraz daha filexible. Ridge'in varyansı teorik olarak daha düşük.

x=model.matrix(Salary~.,Hitters)[,-1]
y=Hitters$Salary

# Ridge Regression
# CV'ı en iyi lambda değerini bulmak için kullanılacak! 
#lambda sonsuz olursa tüm değişkenleri/feature ları sıfırlar/değeri azalır. constant kalır. ? emin değilim.
#lambda=0 olursa normal LR yapar.

library(glmnet)

#alpha=0 - ridge; 1 - lasso demek.

grid=10^seq(10,-2,length=100)  # 100 değişik lambda değeri denesin diye grid  yarattık.
ridge.mod=glmnet(x,y,alpha=0,lambda=grid) #
dim(coef(ridge.mod))
ridge.mod$lambda[50] #  lambda= 11497.57
coef(ridge.mod)[,50] # lambda=11497.57 iken çıkan coefficient değerleri. 
sqrt(sum(coef(ridge.mod)[-1,50]^2)) # MSE.
ridge.mod$lambda[60]
coef(ridge.mod)[,60]
sqrt(sum(coef(ridge.mod)[-1,60]^2))
predict(ridge.mod,s=50,type="coefficients")[1:20,]

set.seed(1)
train=sample(1:nrow(x), nrow(x)/2)
test=(-train)
y.test=y[test]
ridge.mod=glmnet(x[train,],y[train],alpha=0,lambda=grid, thresh=1e-12)
ridge.pred=predict(ridge.mod,s=4,newx=x[test,])
mean((ridge.pred-y.test)^2)
mean((mean(y[train])-y.test)^2)
ridge.pred=predict(ridge.mod,s=1e10,newx=x[test,])
mean((ridge.pred-y.test)^2)
ridge.pred=predict(ridge.mod,s=0,newx=x[test,],exact=T)
mean((ridge.pred-y.test)^2)
lm(y~x, subset=train)
predict(ridge.mod,s=0,exact=T,type="coefficients")[1:20,]

## CV for Ridge and Lasso:
set.seed(1)
cv.out=cv.glmnet(x[train,],y[train],alpha=0) # otomatik bir 10 fold CV. 
plot(cv.out) # lambda değerine göre MSE grafiği.
bestlam=cv.out$lambda.min
bestlam
ridge.pred=predict(ridge.mod,s=bestlam,newx=x[test,])
mean((ridge.pred-y.test)^2)
out=glmnet(x,y,alpha=0)
predict(out,type="coefficients",s=bestlam)[1:20,]

# The Lasso

lasso.mod=glmnet(x[train,],y[train],alpha=1,lambda=grid)
plot(lasso.mod)
set.seed(1)
cv.out=cv.glmnet(x[train,],y[train],alpha=1)
plot(cv.out)
bestlam=cv.out$lambda.min
lasso.pred=predict(lasso.mod,s=bestlam,newx=x[test,])
mean((lasso.pred-y.test)^2)
out=glmnet(x,y,alpha=1,lambda=grid)
lasso.coef=predict(out,type="coefficients",s=bestlam)[1:20,]
lasso.coef
lasso.coef[lasso.coef!=0]


# Chapter 6 Lab 3: PCR and PLS Regression

# Principal Components Regression
# Eugen vektörlere göre ... 
#PCR:Avantajı: her feature'dan birazcık bilgi alıyor feature'u direkt atmıyor.
#PCR: Eğer feature atılmas ıgerekiyorsa PCR kötü sonuç verebilir.
#PLS-supervised-: Lasso gibi çalışıyor. Dependent variable verisini de kullanıyor. (PCA + regression with dependent)

library(pls)

set.seed(2)
pcr.fit=pcr(Salary~., data=Hitters,scale=TRUE,validation="CV")
summary(pcr.fit)
validationplot(pcr.fit,val.type="MSEP")
set.seed(1)
pcr.fit=pcr(Salary~., data=Hitters,subset=train,scale=TRUE, validation="CV")
validationplot(pcr.fit,val.type="MSEP")
pcr.pred=predict(pcr.fit,x[test,],ncomp=7)
mean((pcr.pred-y.test)^2)
pcr.fit=pcr(y~x,scale=TRUE,ncomp=7)
summary(pcr.fit)

# Partial Least Squares

set.seed(1)
pls.fit=plsr(Salary~., data=Hitters,subset=train,scale=TRUE, validation="CV")
summary(pls.fit)
validationplot(pls.fit,val.type="MSEP")
pls.pred=predict(pls.fit,x[test,],ncomp=2)
mean((pls.pred-y.test)^2)
pls.fit=plsr(Salary~., data=Hitters,scale=TRUE,ncomp=2)
summary(pls.fit)


#okuma chapter 8 !

Chapter 6: In-Class Question

In this  exercise, we will predict  the number  of applications received
using the other variables  in the College data  set.


(a)  Split the data set into a training set and a test set.

(b)  Fit a linear model using least squares on the training set, and report the test error obtained.

(c)  Fit a ridge regression model on the training set, with lambda chosen by cross-validation. Report the test error obtained.

(d)  Fit a lasso model on the training set, with lambda chosen by cross-validation. Report the test error obtained, along with the number of non-zero coeffi?cient estimates.

(e)  Fit a PCR  model on the training set, with M chosen by cross-validation. Report the test error obtained, along with the value of M selected by cross-validation.

(f)  Fit a PLS model on the training set, with M chosen by cross-validation. Report the test error obtained, along with the value of M selected by cross-validation.

(g)  Comment on the  results  obtained. How accurately can we predict the number of college applications received? Is there much diff?erence among the test errors resulting from these approaches?