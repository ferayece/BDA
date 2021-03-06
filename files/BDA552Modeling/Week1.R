library(MASS)
library(ISLR)
options(scipen=999) #scintific notation kullanma demek.e li yaz�m� kald�r�r.

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
  #Residuals: hata k�sm�. (y=a + bx + E = b0 + b1x + E)
  #Coefficients: 
    #Intercept= b0 = 7, ... 
    #Slope/e�im of TV= 0.04   --> salesi Tv pozitif etkiliyor.
names(lm.fit)
coef(lm.fit) # direkt denklemi ��karmak istiyorsan. 
confint(lm.fit) #confidence interval � verir. Ne zaman kullan�r�m? Tahminleme verirken %95 olas�l�kla min 10-max 15 diyebilirsin. 
# Ba�ka ne i�e yarar? Bu coefficient istatistiksel olarak 0'dan farkl� m� de�ilmi ? 
#p de�erine bakars�n! %5 den k���kse bunu reject et. P value'nun hipotezi bu s�f�ra e�it.  
#Interceptin p value'suna �ok tak�lm�yoruz.

#predict with fitted model:
#confidence interval: daha geni�; genelde bu kullan�l�yor.
#prediction: daha az geni�.
predict(lm.fit,data=Advertising$TV,interval = "confidence")

#TV=100 de�eri i�in sales nolur?
predict(lm.fit,newdata=data.frame(TV=100),interval = "confidence")

#Plot Tv vs sales column and fitted line from LR:
plot(Advertising$TV,Advertising$sales)
#abline(lm.fit)
abline(lm.fit,lwd=3,col="red") #kal�nl�k renk

##############
#Multiple Linear Regression: 3 column u da predictor olarak al�yoruz.

lm.fit=lm(sales~TV+radio+newspaper,data=Advertising)   
#lm(sales~.,data=Advertising)   
summary(lm.fit)  #Adjusted R-squared:  0.8956
# Tv coef. � 0.047 den 0.045 e d��t�. Robost imi�. Bu kararl� bir predictor. -dominant� s�yleyemeyiz burdan.
#newspaper'�n p value'su 0.05 in �st�nde. Statistically anlaml� bir predictor de�il. Bunu ��karal�m.

lm.fit2=lm(sales~.-newspaper,data=Advertising) 
#lm(sales~TV+radio,data=Advertising) 
summary(lm.fit2)   #Adjusted R-squared:  0.8962 

#iki modeli kar��la�t�r�rken Adjusted R-Squared'a bak�yoruz. 
#Ad. R-Squared:Model, Toplam varyans�n ka�ta ka��n� a��klar? �n cevab�.
#Adjusted: her variable da her de�i�kene penalty ile adjusted eder.
#R square: her variable da artar ama adjusted R-square penalty ile bunu dengeler. O y�zden adjusted'a bakar�z.
#Adj R-square artarsa modelim daha iyi. 

#F-statistic:
#100 predictor ekledim, hepsi de ��p. Buna ra�men en az 5 tanesini significant oldu�unu g�rebilirim.
#P-value �ok fazla predictor'de do�ruyu vermeyebilir. (%5 do�ruluk ile) 100 de�i�ken koyarsam 95 inde do�ru 5 in de yanl�� sonu� verir.
#�ok predictor koyunca F-statistic'e de bakmal�y�z. F-test b�t�n de�i�kenler i�inde en az 1'i significant m� yoksa hepsi ��p m�?
# F-statistic > 1 ise p-value'ya bakar�z 0.05'den k���k m�? K���kse en az 1'i anlaml�. Chapter 3'� oku!!!

#install.packages("rgl")
#install.packages("car")
#scatter3d(y=,x=,z=)

###################

#Categorical Variables:

summary(Credit) # LSR i�inde gelen bir dataset.

lm.fit=lm(Balance~Gender,data = Credit)
summary(lm.fit)  # Gender �n credit �st�nde bi etkisi yok.

lm.fit=lm(Balance~Ethnicity,data = Credit)
summary(lm.fit)  # Ethnicity �n credit �st�nde bi etkisi yok. (p-value ya bak aq)

#Interaction Terms

#maas credit balance � etkiler mi? Bir adam�n student olmas�? Ayr� ayr� bak�yoruz!!!
lm.fit1=lm(Balance~Income+Student,data = Credit)
summary(lm.fit1) #ayr� ayr� bu predictor'lerin etkisini g�rd�k.

#�imdi d���k gelirli ��rencilerin credit score'u? INTERACTION TERM.�NCOME:STUDENT =INTERACTION TERM
lm.fit2=lm(Balance~Income+Student+Income:Student,data = Credit)
summary(lm.fit2)
#�NTERACT�ONNTERM P-VALUE >0.05; anlams�zm��. 
#yani �ncome ve student aras�nda anlaml� bir ili�ki yokmu�. Ayr� ayr� kullan�ls�n.Intr. term� at gitsin.
#ayr� ayr� anlams�z ama interaction term'� anlaml� ��karsa kural hepsini tutmak! 

####################
#Non-linear Relationship:
AutoData <- Auto
summary(Auto)

lm.fit= lm(mpg~horsepower,data=Auto)
summary(lm.fit)
plot(lm.fit) 
#liner olmad���n� g�rd�k. Lineer olsa residual grafi�i a�a�� inmeli direkt ama biraz yukar� ��km��.
#normal Q-Q: residual ler normal da��l�yorsa buras� d�z olur. Bu �rnekte normal dist.
#Scale Location: 
#Resid vs Leverage: High Leverage point var ise denklem s�k�nt�. bunlar� at�p �al��t�rmak daha mant�kl�. 
lm.fit1=lm(mpg~horsepower + I(horsepower^2),data = Auto)
summary(lm.fit1)
plot(lm.fit1)

###########################Classification###############################

#Parametric Classifcation:
  # dependent variable=categorical ! Yani it s a classification problem.
  # 1.Logistic Regression - iki class birbirinden �ok iyi ayr�lm�ssa logistic bunu �ok iyi yakalayam�yor.
  # 2. Linear Discriminant Analysis: iki class �ok iyi ayr�ld�ysa iyi �al���r.Assumption: t�m predictorlar/exploraty variable'lar normal dag�l�r(gaussian a uyar).
    # Covariance matrix her bir predictor i�in ayn� assmption� da var. Bu ikisi tutarsa iyi �al���r.
  # 3. Quadratic Discriminant Analysis aspp: yukar�daki ile ayn� ama covariance matrix her predictor i�in farkl� diyor.
# ger�ek hayatta asmp lar �ok �nemli de�il, kar��la�t�rmak en iyisi.
#Parametric modellerde assmp'lar inference i�in kullan�yoruz. 
#the stock market data: depend variable fiyat artar m� azal�r m�? = categorical ! Yani classification problem.
attach(Smarket)  # env. variable olarak y�kledik. 
names(Smarket)   #paketten geldi.

plot(Volume)
plot(Direction) #dependent variable.

#logistic regression:
glm.fit=glm(Direction~Lag1+Lag2+Lag3+Lag4+Lag5+Volume,data=Smarket,family=binomial)   #binomial=logistic
summary(glm.fit) # p value lara bak�nca ��pp.
