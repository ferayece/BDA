setwd("C:/Users/Student/Downloads")
load("descr.rdata")
load("mutagen.rdata")
descr<-descr[,1:100] #çok büyük data o yüzden ilk 100 feature u aliyoruz.
library("caret") 
library("FactoMineR")
library("e1071")
set.seed(1)
inTrain <- createDataPartition(mutagen, p = 3/4, list = FALSE) #split dataset as train/test. default stratified sampling yapiyor
#Strafied sampling: Class imbalance varsa bunu dikkate alan sampling. %55 mutagen ise seçtigin sample da da böyledir bu oran.
trainDescr <- descr[inTrain,]
testDescr <- descr[-inTrain,]
trainClass <- mutagen[inTrain]
testClass <- mutagen[-inTrain]
prop.table(table(mutagen)) #oranlar.
prop.table(table(trainClass))

ncol(trainDescr)
descrCorr <- cor(trainDescr)
highCorr <- findCorrelation(descrCorr, 0.90) #corelasyonu %90 üzeri olanlari attik. iki degiskeni de atiyor !!! iki degisken arasindaki korelasyon yuksekse at.
#Çok az degiskende highly correlated predictorler zararsiz ama cok feature'da zararli. overfite maal verir.
#Diment. reduction yapabilirz ama variable kaybetmek istemiyorum. Factorleri alirsam variable isimlerini kaybedicem, bunu istemedim.
#factoru anlamlandirmak icin,predictorler ile iliskisine bakip hangisi en yuksek korele ise bu variable dominanttir diyosun. varsayimsal.
#bu atma islemini tek tek yapmak daha dogrudur. 
trainDescr <- trainDescr[, -highCorr]
testDescr <- testDescr[, -highCorr]
ncol(trainDescr)

###preprocessing 
#SVM, PCA,Neural network için kesinlikle normalizasyon gerekir!
xTrans <- preProcess(trainDescr) #caretin hizli normalize etme func.
trainDescr <- predict(xTrans, trainDescr)
testDescr <- predict(xTrans, testDescr)

#kumulatif olarak varyansin %80-85'ini aciklayacak kadar vector yaratmali.
pca3 = PCA(trainDescr, graph = FALSE)
pca3$eig
pca3 = PCA(trainDescr, ncp=14, graph = FALSE)
pca<-pca3$ind$coord

###preprocessing bitti.

#https://topepo.github.io/caret/ CARETIN USERGUIDE'I!

#bootstrapping:with replacement train set -  otomatik evaluation,i caretin.
#gridSearch 
#bootControl <- trainControl(number = 10) # 10 degisik train set ol
#usturcak ve denicek. SADECE 10 KERE BOOTSTRAPPING YAP.  
set.seed(2) 

gbmGrid <-  expand.grid(interaction.depth = c(1, 5, 9), 
                        n.trees = c(1,5,20), 
                        shrinkage = 0.1,
                        n.minobsinnode = 20)

#trainDescr== predictors
#trainClass == dependent variable
#tuneGrid hangi grid ile deneyecek?
#trControl defaultu bootstrap. bootControl kendimiz yukarida olusturduk. 10 tane bootstrap yap dedik.

gbmFit <- train(trainDescr, trainClass, method = "gbm", 
                trControl = bootControl, verbose = FALSE, 
                bag.fraction = 0.5, tuneGrid = gbmGrid)

plot(gbmFit) #boosting iterations  --> n.trees parametresi aslinda. Bu bir bug ! aklini karistirma.
#classification cozdugumuz icin accuracy'ye bakicaz. yukaridaki grapha göre; 20 interaction node, 9 (20-9 secilecek)
plot(gbmFit, plotType = "level")
gbmFit$results

#en yuksek 2, en az kompleks i seç. 9 u burdan da secebiliriz.
best(gbmFit$results, metric="Accuracy", maximize=T)
tolerance(gbmFit$results, metric="Accuracy", maximize=T, tol=2)

#bunla da best accuracy yi veren kombinasyonu bulabiliriz: booststrap yerine 10-K CV yapti. Her bir olasilik icin 10 kere CV yapti; ortalamayi aldi.
fitControl <- trainControl(## 10-fold CV
  method = "repeatedcv",
  number = 10,
  ## repeated ten times : totalde 10*10 100 deneme.
  repeats = 1)

# parametreyi nasil fit edecegine dair fikrin yoksa bu fitControl gibi bisi kullanman lazim. 
gbmFit <- train(trainDescr, trainClass, method = "gbm", 
                trControl = fitControl, verbose = FALSE, 
                tuneLength = 4) # interaction.depth,,n.trees vs parametreleri için 4 farkli deger deneyecek.

plot(gbmFit)                       
plot(gbmFit, plotType = "level")
gbmFit$results  

best(gbmFit$results, metric="Accuracy", maximize=T)
tolerance(gbmFit$results, metric="Accuracy", maximize=T, tol=2)

gbmFit$results[7,] # en iyi tolerance 7 çikti o yuzden 7.row u aldik.  

##botstrapping de caret pakedini kullanmayi bitirdik. Metod: grid seç, çalistir. 

#XGoost: Extreme boosting. 
#boosting anlar XGBoost cat. variable anlamaz.Onehot encoding 
#one-hot encoding: dummy variable olusturmanin havali adi. Categirocal variable dan category kader kolon olusturma.
#XGBoost categorical variable kabul etmez.
#100 kategori varsa 99 kolon olusturusun. (sonuncusu 99 dan biri degildir mantigi.)
library(data.table)
library(mlr)

setcol<-c("age","workclass","fnlwgt","education","education-num",
          "marital-status","occupation","relationship","race","sex",
          "capital-gain", "capital-loss","hours-per-week","native-country",
          "target")

train <- read.table("adulttrain.txt", header = F, 
                    sep = ",", col.names = setcol, na.strings = c(" ?"),
                    stringsAsFactors = F)

test <- read.table("adulttest.txt",header = F,sep = ",",col.names = setcol,
                  na.strings = c(" ?"),stringsAsFactors = F)

setDT(train)  #data table a cevir.
setDT(test)

table(is.na(train))
sapply(train, function(x) sum(is.na(x))/length(x))*100 # her kolon icin mean leri hesapliyor.

table(is.na(test))
sapply(test, function(x) sum(is.na(x))/length(x))*100

#View(train) 
library(stringr)
library(mlr)
library(data.table)

#trim:
test [,target:=substr(target,start = 1,stop = nchar(target)-1)]
char_col <- colnames(train)[ sapply (test,is.character)]
for(i in char_col) set(train,j=i,value = str_trim(train[[i]],side = "left"))
for(i in char_col) set(test,j=i,value = str_trim(test[[i]],side = "left"))

train[is.na(train)] <- "Missing" #null'lari missing atadik.
test[is.na(test)] <- "Missing"

##onehot encoding.
new_tr <- model.matrix(~.+0,data=train[,-c("target"),with=F]) #onehot encoding.
new_ts <- model.matrix(~.+0,data = test[,-c("target"),with=F])

k=1
for (i in colnames(new_tr)) {
  check<-(i %in% colnames(new_ts))
  print(check)
  print(k)
  k=k+1
}

#traindeki bi kategori, testte yok datayi stratified yapmamislar datada böyle.
#o yuzden bi tanesini trainden cikardik. (yukaridaki for dan false döneni)
new_tr<-new_tr[,-74]

#50K altina 1, 50K ustune 0 veriyorum target.
labels <- train$target
ts_label <- test$target
labels<-(labels=="<=50K")
labels<-as.numeric(labels)
ts_label<-(ts_label=="<=50K")
ts_label<-as.numeric(ts_label)

#buraya kadar xgboost a input olacak datayi hazirladik.
library("xgboost")

#hazirlanan datayi Dmatrix e çevirelim.
dtrain <- xgb.DMatrix(data = new_tr,label = labels)
dtest <- xgb.DMatrix(data = new_ts,label=ts_label)

params <- list(booster = "gbtree", objective = "binary:logistic", #classification logistic; regression = gaussian
               eta=0.3, #learning rate, shrinkage.
               gamma=0, #penalty parametrem. tree regularization pm for avoiding overfit.
               max_depth=6,
               min_child_weight=1,#overfiti engelleyen bi pm 
               subsample=1, 
               colsample_bytree=1) # her sferinde feature larin kaçini kullanacak? 1=hepsini.

#en iyi iterasyon hangisi? CV ile bul!
xgbcv <- xgb.cv( params = params, data = dtrain, 
                 nrounds = 100, 
                 nfold = 5, # 5 fold CV 
                 showsd = T, 
                 stratified = T, 
                 print_every_n = 10, 
                 early_stopping_rounds = 20, 
                 maximize = F)

xgbcv$best_iteration

xgb1 <- xgb.train (params = params, data = dtrain, 
                   nrounds = xgbcv$best_iteration)

xgbpred <- predict (xgb1,dtest)
xgbpred <- ifelse (xgbpred > 0.5,1,0)

table(xgbpred, ts_label)

mat <- xgb.importance (feature_names = colnames(new_tr),model = xgb1)
xgb.plot.importance (importance_matrix = mat[1:20])


##caret pakedi ile XGBoost:
xgbGrid <-  expand.grid(nrounds = c(10,100), #try to optimze it.
                        max_depth = c(5, 10), #try to optimze it.
                        eta = 0.3,
                        gamma = 0, colsample_bytree=1,
                        min_child_weight=1, subsample=1)

fitControl <- trainControl( ## 10-fold CV
  method = "repeatedcv",
  number = 2,  ##2-fold CV 
  ## repeated ten times
  repeats = 1)

gbmFit <- train(new_tr, as.factor(labels), method = "xgbTree", 
                trControl = fitControl, verbose = T, 
                tuneGrid = xgbGrid)

plot(gbmFit)                       
plot(gbmFit, plotType = "level")
gbmFit$results

best(gbmFit$results, metric="Accuracy", maximize=T)
tolerance(gbmFit$results, metric="Accuracy", maximize=T, tol=2)


# yeni gelistirilen method: Light gradient boosting. Son versiyon! 
#yüklemek için: https://github.com/Microsoft/LightGBM/tree/master/R-package
#ama yuklemesi zor. 
# https://github.com/Laurae2/lgbdl/
#microsoft yazdigi icin visual studio yu yüklememiz gerekiyor. 
# leaf lere odaklanip leaf leaf ilerliyor.
#büyük veri setleri için optimize edilmistir! 10bin+ obs. için iyi çalisiyor. 