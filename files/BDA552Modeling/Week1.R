library(MASS)
library(ISLR)
options(scipen=999) #scintific notation kullanma demek.e li yazýmý kaldýrýr.

# Set the directory:
setwd("C:/Users/TOSH/Desktop/Modeling&Validation")

# Load Data:
Advertising <- read.csv2("Advertising.csv",stringsAsFactors = F,sep=",",dec=".")
Advertising=Advertising[,-1] # sales: dependent variable.

#build simple LR: with just TV column (LR: parametric model!)
names(Advertising)
lm.fit=lm(sales~TV,data=Advertising) #fit Advertising Data

summary(lm.fit) 
#output okuma:
  #Residuals: hata kýsmý. (y=a + bx + E = b0 + b1x + E)
  #Coefficients: 
    #Intercept= b0 = 7, ... 
    #Slope/eðim of TV= 0.04   --> salesi Tv pozitif etkiliyor.
names(lm.fit)
coef(lm.fit) # direkt denklemi çýkarmak istiyorsan. 
confint(lm.fit) #confidence interval ý verir. Ne zaman kullanýrým? Tahminleme verirken %95 olasýlýkla min 10-max 15 diyebilirsin. 
# Baþka ne iþe yarar? Bu coefficient istatistiksel olarak 0'dan farklý mý deðilmi ? 
#p deðerine bakarsýn! %5 den küçükse bunu reject et. P value'nun hipotezi bu sýfýra eþit.  
#Interceptin p value'suna çok takýlmýyoruz.

#predict with fitted model:
#confidence interval: daha geniþ; genelde bu kullanýlýyor.
#prediction: daha az geniþ.
predict(lm.fit,data=Advertising$TV,interval = "confidence")

#TV=100 deðeri için sales nolur?
predict(lm.fit,newdata=data.frame(TV=100),interval = "confidence")

#Plot Tv vs sales column and fitted line from LR:
plot(Advertising$TV,Advertising$sales)
#abline(lm.fit)
abline(lm.fit,lwd=3,col="red") #kalýnlýk renk

##############
#Multiple Linear Regression: 3 column u da predictor olarak alýyoruz.

lm.fit=lm(sales~TV+radio+newspaper,data=Advertising)   
#lm(sales~.,data=Advertising)   
summary(lm.fit)  #Adjusted R-squared:  0.8956
# Tv coef. ý 0.047 den 0.045 e düþtü. Robost imiþ. Bu kararlý bir predictor. -dominantý söyleyemeyiz burdan.
#newspaper'ýn p value'su 0.05 in üstünde. Statistically anlamlý bir predictor deðil. Bunu çýkaralým.

lm.fit2=lm(sales~.-newspaper,data=Advertising) 
#lm(sales~TV+radio,data=Advertising) 
summary(lm.fit2)   #Adjusted R-squared:  0.8962 

#iki modeli karþýlaþtýrýrken Adjusted R-Squared'a bakýyoruz. 
#Ad. R-Squared:Model, Toplam varyansýn kaçta kaçýný açýklar? ýn cevabý.
#Adjusted: her variable da her deðiþkene penalty ile adjusted eder.
#R square: her variable da artar ama adjusted R-square penalty ile bunu dengeler. O yüzden adjusted'a bakarýz.
#Adj R-square artarsa modelim daha iyi. 

#F-statistic:
#100 predictor ekledim, hepsi de çöp. Buna raðmen en az 5 tanesini significant olduðunu görebilirim.
#P-value çok fazla predictor'de doðruyu vermeyebilir. (%5 doðruluk ile) 100 deðiþken koyarsam 95 inde doðru 5 in de yanlýþ sonuç verir.
#Çok predictor koyunca F-statistic'e de bakmalýyýz. F-test bütün deðiþkenler içinde en az 1'i significant mý yoksa hepsi çöp mü?
# F-statistic > 1 ise p-value'ya bakarýz 0.05'den küçük mü? Küçükse en az 1'i anlamlý. Chapter 3'ü oku!!!

#install.packages("rgl")
#install.packages("car")
#scatter3d(y=,x=,z=)

###################

#Categorical Variables:

summary(Credit) # LSR içinde gelen bir dataset.

lm.fit=lm(Balance~Gender,data = Credit)
summary(lm.fit)  # Gender ýn credit üstünde bi etkisi yok.

lm.fit=lm(Balance~Ethnicity,data = Credit)
summary(lm.fit)  # Ethnicity ýn credit üstünde bi etkisi yok. (p-value ya bak aq)

#Interaction Terms

#maas credit balance ý etkiler mi? Bir adamýn student olmasý? Ayrý ayrý bakýyoruz!!!
lm.fit1=lm(Balance~Income+Student,data = Credit)
summary(lm.fit1) #ayrý ayrý bu predictor'lerin etkisini gördük.

#Þimdi düþük gelirli öðrencilerin credit score'u? INTERACTION TERM.ýNCOME:STUDENT =INTERACTION TERM
lm.fit2=lm(Balance~Income+Student+Income:Student,data = Credit)
summary(lm.fit2)
#ÝNTERACTÝONNTERM P-VALUE >0.05; anlamsýzmýþ. 
#yani ýncome ve student arasýnda anlamlý bir iliþki yokmuþ. Ayrý ayrý kullanýlsýn.Intr. termü at gitsin.
#ayrý ayrý anlamsýz ama interaction term'ü anlamlý çýkarsa kural hepsini tutmak! 

####################
#Non-linear Relationship:
AutoData <- Auto
summary(Auto)

lm.fit= lm(mpg~horsepower,data=Auto)
summary(lm.fit)
plot(lm.fit) 
#liner olmadýðýný gördük. Lineer olsa residual grafiði aþaðý inmeli direkt ama biraz yukarý çýkmýþ.
#normal Q-Q: residual ler normal daðýlýyorsa burasý düz olur. Bu örnekte normal dist.
#Scale Location: 
#Resid vs Leverage: High Leverage point var ise denklem sýkýntý. bunlarý atýp çalýþtýrmak daha mantýklý. 
lm.fit1=lm(mpg~horsepower + I(horsepower^2),data = Auto)
summary(lm.fit1)
plot(lm.fit1)

###########################Classification###############################

#Parametric Classifcation:
  # dependent variable=categorical ! Yani it s a classification problem.
  # 1.Logistic Regression - iki class birbirinden çok iyi ayrýlmýssa logistic bunu çok iyi yakalayamýyor.
  # 2. Linear Discriminant Analysis: iki class çok iyi ayrýldýysa iyi çalýþýr.Assumption: tüm predictorlar/exploraty variable'lar normal dagýlýr(gaussian a uyar).
    # Covariance matrix her bir predictor için ayný assmptioný da var. Bu ikisi tutarsa iyi çalýþýr.
  # 3. Quadratic Discriminant Analysis aspp: yukarýdaki ile ayný ama covariance matrix her predictor için farklý diyor.
# gerçek hayatta asmp lar çok önemli deðil, karþýlaþtýrmak en iyisi.
#Parametric modellerde assmp'lar inference için kullanýyoruz. 
#the stock market data: depend variable fiyat artar mý azalýr mý? = categorical ! Yani classification problem.
attach(Smarket)  # env. variable olarak yükledik. 
names(Smarket)   #paketten geldi.

plot(Volume)
plot(Direction) #dependent variable.

#logistic regression:
glm.fit=glm(Direction~Lag1+Lag2+Lag3+Lag4+Lag5+Volume,data=Smarket,family=binomial)   #binomial=logistic
summary(glm.fit) # p value lara bakýnca çöpp.
