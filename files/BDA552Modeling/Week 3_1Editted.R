# Chapter 8 Lab: Decision Trees

#Fitting Classification Trees

#Loading packages
library(tree)
library(ISLR)
attach(Carseats)
?Carseats

#Forming classfying data Classification problem �na �evir!
High=ifelse(Sales<=8,"No","Yes")

#Attaching new variable to the data frame
Carseats=data.frame(Carseats,High)

#Rpart, rattle DT'de prob g�rme k�t�phaneleri.

#Forming the first decision tree
tree.carseats=tree(High~.-Sales,Carseats)
#tree.carseats 
summary(tree.carseats)

#plot decsision tree
plot(tree.carseats) # bok gibi ��kt� ba�l�klar yok anlamad�m.
text(tree.carseats,pretty=0) # ba�l�klar burdan ekleniyormu� :)

#show decision tree branches
tree.carseats

#Create validation set
set.seed(2)
train=sample(1:nrow(Carseats), 200) #train index
Carseats.test=Carseats[-train,] #test index
High.test=High[-train] #dependent variable � train test diye b�ld�k.

#Forming the decision tree using training set
tree.carseats=tree(High~.-Sales,Carseats,subset=train)

#Predict using validation set
tree.pred=predict(tree.carseats,Carseats.test,type="class")
table(tree.pred,High.test) # pred,actual
#Accuracy=(86+57)/200

# Number of final node hesab� i�in CV yapal�m. Hangi seviye pruning yapaca��m�z� bulmak i�in CV!
#Form tree using 10-fold cross validating and pruning
set.seed(3)
#We use the argument FUN=prune.misclass in order to indicate that we want the
#classiﬁcation error rate to guide the cross-validation and pruning process,
#rather than the default for the cv.tree() function, which is deviance. 
cv.carseats=cv.tree(tree.carseats,FUN=prune.misclass) 
#cv.tree; bu i� i�in kullan�l�yor. misclassify a g�re s�rala hangi node'un final node olaca��n� bulmak i�in.
names(cv.carseats)
cv.carseats
#train dataset e CV yaparak ka� tane terminal node olmas� gerekti�ini bulduk.
#dev: 48 =min ; size=9 (48'in kar��l���.) 9 tane terminal node olmas� gerekti�ini s�yledi. Bunu parametre olarak vericez tree'ye.
par(mfrow=c(1,2))
plot(cv.carseats$size,cv.carseats$dev,type="b")
plot(cv.carseats$k,cv.carseats$dev,type="b")

#Form tree with 9 terminal nodes
prune.carseats=prune.misclass(tree.carseats,best=9) #elle giricez buldu�umuz de�eri yukar�da.
plot(prune.carseats)
text(prune.carseats,pretty=0) #pretty=1 baz� bilgileri gizler.
tree.pred=predict(prune.carseats,Carseats.test,type="class")
table(tree.pred,High.test)
#Accuracy: (94+60)/200

#Form tree with 15 terminal nodes
prune.carseats=prune.misclass(tree.carseats,best=15)
plot(prune.carseats)
text(prune.carseats,pretty=0)
tree.pred=predict(prune.carseats,Carseats.test,type="class")
table(tree.pred,High.test)
(86+62)/200

#Fitting Regression Trees
library(MASS)
set.seed(1)
train = sample(1:nrow(Boston), nrow(Boston)/2)
tree.boston=tree(medv~.,Boston,subset=train)
summary(tree.boston)
plot(tree.boston)
text(tree.boston,pretty=0)

#Fitting Regression Trees using 10-fold cv
cv.boston=cv.tree(tree.boston)
cv.boston   # en iyi 8 ��kt�.
plot(cv.boston$size,cv.boston$dev,type='b')

#Fitting Regression Trees using 4 terminal nodes
prune.boston=prune.tree(tree.boston,best=8)  #best=4 denedik.
plot(prune.boston)
text(prune.boston,pretty=0)

#Fitting Unpruned Regression Tree
yhat=predict(tree.boston,newdata=Boston[-train,])
boston.test=Boston[-train,"medv"] #medv predict etmeye �al��t���m�z kolon.
#plot(yhat,boston.test) actual pred grafi�i.
abline(0,1)
mean((yhat-boston.test)^2)

#Fitting Pruned Regression Tree
#yhat=predict(tree.boston,newdata=Boston[-train,])
yhat=predict(prune.boston,newdata=Boston[-train,])
boston.test=Boston[-train,"medv"] #medv predict etmeye �al��t���m�z kolon.
#plot(yhat,boston.test) actual pred grafi�i.
abline(0,1)
mean((yhat-boston.test)^2)

#Fitting Bagging
#Art�k treeleri g�remiyoruz ��nk� ortalama bir model veriyor bize. Importance ile �nemli predictorleri g�rcez.
#DT interpretable ike nbu metodlaar de�il. importance=TRUE hangi predictor daha �nemli bize ver diyoruz.
#randomForest ile bagging ayni sey; mtry yi bos birakirsan randomForest olur. 
library(randomForest)
set.seed(1)
bag.boston=randomForest(medv~.,data=Boston,subset=train,mtry=13,importance=TRUE) #bagging.
bag.boston # summary of RandomForest.
# Mean of squared residuals:  variance �n ka�ta ka��n� a��kl�yor? Burada 11.157 ��kt�.
#mtry=13 --> No. of variables tried at each spli

#Predicting validation set
yhat.bag = predict(bag.boston,newdata=Boston[-train,])
plot(yhat.bag, boston.test)
abline(0,1)
mean((yhat.bag-boston.test)^2) #MSE ka�?  13.50. D��t�.
#MSE yi k���ltmek �nemli! Baging DT'den daha k���k bir MSE yaratt�. Good.

#Change the number of trees grown
bag.boston=randomForest(medv~.,data=Boston,subset=train,mtry=13,ntree=25) # ntree=25, ka� tane bootstrap yapaca��/tree olu�turaca��? 25 tane tree nin ortalamas�n� al dedik.D���rmemen laz�m.
yhat.bag = predict(bag.boston,newdata=Boston[-train,])
mean((yhat.bag-boston.test)^2)  

#Fitting random forest
set.seed(1)
rf.boston=randomForest(medv~.,data=Boston,subset=train,mtry=6,importance=TRUE)
yhat.rf = predict(rf.boston,newdata=Boston[-train,])
mean((yhat.rf-boston.test)^2)
importance(rf.boston) #treeleri g�remedi�imiz i�in relative importance lar� g�relim.
#importance ne kadar y�ksekse o kadar iyi.
varImpPlot(rf.boston)
#IncMSE grafi�i en �nemliden en a�a��ya diziyor. rm ��akrd���m�zda MSE ne kadar artacak diyerek onu en �se koyuyor.

#Boosting
library(gbm)
set.seed(1)
boost.boston=gbm(medv~.,data=Boston[train,],
                 distribution="gaussian", #reg:gaussian, clasf: bernoulli  medv cont de�er.
                 n.trees=5000,  # ka� tree �reticcez?
                 shrinkage=0.2, # 0-1 aras�nda. ne kadar k���k olursa o kadar yava� error fit yap�yor. ne kadar k���k
                 #olursa (yava� ��renirse detayl� ��renir) o kadar overfite yatk�n olur.
                 interaction.depth=4) #ka� terminal node olsun en son?
summary(boost.boston)

#marginal effects of variables
par(mfrow=c(1,2))
plot(boost.boston,i="rm")
plot(boost.boston,i="lstat")

#predicting using validation set
yhat.boost=predict(boost.boston,newdata=Boston[-train,],n.trees=5000)
mean((yhat.boost-boston.test)^2) #MSE ka�? 12.14 d��t� bagging ve DT ye g�re.

boost.boston=gbm(medv~.,data=Boston[train,],distribution="gaussian",n.trees=5000,interaction.depth=4,shrinkage=0.2,verbose=F)
yhat.boost=predict(boost.boston,newdata=Boston[-train,],n.trees=5000)
mean((yhat.boost-boston.test)^2)

#cross validation


#Manuel CV: without replacement.10 fold.
set.seed(1)
#Randomly shuffle the data
cvBoston<-Boston[sample(nrow(Boston)),] #boston datasetini shuffle ettik.
folds <- cut(seq(1,nrow(cvBoston)),breaks=10,labels=FALSE) # 10 a b�l�yoruz. 


mse<-rep(NA,10)
shrink <- c(0.001,0.01,0.05,0.1,0.15,0.2)

for (j in shrink) {
  for(i in 1:10){
    #set.seed(1)
    #Segement your data by fold using the which() function 
    testIndexes <- which(folds==i,arr.ind=TRUE)
    testData <- cvBoston[testIndexes, ]
    trainData <- cvBoston[-testIndexes, ]
    
    model=gbm(medv~.,data=trainData,
                           distribution="gaussian",
                n.trees=100,interaction.depth=4,
              shrinkage=j)
    
    yhat = predict(model,newdata=testData,n.trees=100)
    mse[i]<-mean((yhat-testData$medv)^2)
    #Use the test and train data partitions however you desire...
    
  }
  print(mean(mse)) # print shrink + mean de yapabilirsin.
}

#Automatic 10 fold CV:
library(caret)
flds <- createFolds(cvBoston, k = 10, list = F, returnTrain = FALSE)


#######XGBoost -- Di�er dosyadan ekledim. 

#Boostingin geli�tirilmi� hali.
#Regularization var boosting e ek.
#Parallelasyona �ok yatk�n. Ka� core oldu�unu s�yl�yorsunuz o b�l�yor. H�zl� �al���r!
#Data cleaning k�sm� �ok uzun; categorical variable kabul etmez !!! 

library("xgboost")

set.seed(1)
traindata<-as.matrix(Boston[train,-ncol(Boston)]) #xgBoost ,i�in matrix e �evirmeliyiz.
trainlabel<-as.matrix(Boston[train,ncol(Boston),drop=FALSE])

dtrain <- xgb.DMatrix(traindata, label=trainlabel) # XgBoost a input olabilecek bir matrix olu�turuyoruz.

#Xgboost un �ok fazla parametresi var.
params <- list(eta=0.3, #learning rate. 1'e ne kadar yak�nsa o kadar h�zl� converge eder.
               gamma=0, # gamma=5 optimal. Regularization. sonsuza ne kadar yak�nsa coef. s�f�ra yakla��r. gamma=0, regularization yok ! 
               max_depth=6, # optimal i�in CV yapmak laz�m. her bir olu�turulan tree'nin level'�. (depth of the tree)
               min_child_weight=1, # her tree i�in terminal node u ne zaman cut-off yapay�m. Grow etmesini durdurur. (tree splitting i ne zaman durduray�m?)
               subsample= 1,# optimal 0.5-0.8 aras�nda.
               colsample_bytree=1 # ka� tane feature u kullanacak? kitaptan bakabilirsin detay�na ... 
               )

# en iyi number of iteration � bulmak.
bstSparse <- xgb.cv(params=params, data = dtrain, 
                    nrounds = 100, #number of iteration.
                    nfold = 5, # trainset i 5 e b�ld�m.
                    print_every_n = 10, #10 ad�mda bir g�ster ��kt�y�.
                    early_stopping_round = 20 #early stopping CV. e�er MSE'den sonra 20 iterationda daha da d��mediyse iteration � durdur.
                    )

xgb1 <- xgb.train (params = params, data = dtrain, nrounds = 75)

testdata<-as.matrix(Boston[-train,-ncol(Boston)])
testlabel<-as.matrix(Boston[-train,ncol(Boston),drop=FALSE])
dtest <- xgb.DMatrix(testdata, label=testlabel)

yhat.xgboost<-predict(xgb1, dtest)
mean((yhat.xgboost-boston.test)^2)

mat<-xgb.importance(feature_names = colnames(testdata),model=xgb1)
xgb.plot.importance(importance_matrix = mat[1:5])

#params <- list(eta=0.3, gamma=0, max_depth=6, 
#               min_child_weight=1, subsample=1, colsample_bytree=1)

set.seed(1)
train <- sample(nrow(Weekly), nrow(Weekly) / 2)
Weekly$Direction <- ifelse(Weekly$Direction == "Up", 1, 0)
Weekly.train <- Weekly[train, ]
Weekly.test <- Weekly[-train, ]

logit.fit <- glm(Direction ~ . - Year - Today, data = Weekly.train, family = "binomial")
logit.probs <- predict(logit.fit, newdata = Weekly.test, type = "response")
logit.pred <- ifelse(logit.probs > 0.5, 1, 0)
table(Weekly.test$Direction, logit.pred)
(282+11)/545

boost.fit <- gbm(Direction ~ . - Year - Today, data = Weekly.train, distribution = "bernoulli", n.trees = 5000)
boost.probs <- predict(boost.fit, newdata = Weekly.test, n.trees = 5000)
boost.pred <- ifelse(boost.probs > 0.5, 1, 0)
table(Weekly.test$Direction, boost.pred)
(109+166)/545

bag.fit <- randomForest(Direction ~ . - Year - Today, data = Weekly.train, mtry = 6)
bag.probs <- predict(bag.fit, newdata = Weekly.test)
bag.pred <- ifelse(bag.probs > 0.5, 1, 0)
table(Weekly.test$Direction, bag.pred)
(215+79)/545

rf.fit <- randomForest(Direction ~ . - Year - Today, data = Weekly.train, mtry = 2)
rf.probs <- predict(rf.fit, newdata = Weekly.test)
rf.pred <- ifelse(rf.probs > 0.5, 1, 0)
table(Weekly.test$Direction, rf.pred)
(70+222)/545
