#Q1. Please provide a class in Python 3 that will include 3 different functions for rectangular prisms. The
#first function will calculate the surface area of a rectangular prism, the second one will calculate the
#volume of this prism, and the third one should properly print these two calculated values on the console.

class rectangular():
    def __init__(self,a,b,c):
        self.a=a
        self.b=b
        self.c=c
    def area(self):
        return 2*(self.a*self.b + self.b*self.c + self.a*self.c)
    def volume(self):
        return self.a*self.b*self.c
    def print_info(self):
        print ('Area: ', self.area())
        print ('Volume: ',self.volume())

NewRect = rectangular(2,3,4)
NewRect.print_info()

#Q2. You are expected to understand the following class and provide the necessary comments to the given places (annotated with “# ….”).
class ComplexNumber():
    def __init__(self,r = 0,i = 0): # we create an instance, default value of r and i will be zero.
        self.real = r #if there is an r value of instance, it will be self.real now. If not, it will be zero.
        self.imag = i #if there is an i value of instance, it will be self.imag now. If not, it will be zero.

    def getData(self): # defining a function for ComplexNumber class
        print("{0}+{1}j".format(self.real,self.imag)) #generate complex number with values of self paremeters of class.

c1 = ComplexNumber(2,3) #create an instance of ComplexNumber class with two parameters (r and i )
# No output because we did not call any function of created instance.
c1.getData() # Output: 2+3j
c2 = ComplexNumber(5) # #create an instance of ComplexNumber class with one parameters (r)
#  No output because we did not call any function of created instance.
c2.attr = 10 # create a inclusive variable (attr) for c2 instance and assign 10 to it.

#Q3. Please make use of the following Customer class with providing Sarah and Hakeem two different
#accounts and a series of deposits. Please try to cover all the functions and attributes.
class Customer(object):
     def __init__(self, name, balance=0.0):
        self.name = name
        self.balance = balance

     def withdraw(self, amount):
        if amount > self.balance:
            raise RuntimeError('Amount greater than available balance.')
        self.balance -= amount
        return self.balance

     def deposit(self, amount):
        self.balance += amount
        return self.balance

c1 = Customer("Sarah",1000.0)
c2=Customer("Hakeem",121.6)

#print ("remained balance of ",c1.name,c1.withdraw(5))
#print ("remained balance of ",c2.name,c2.deposit(15))

#Q4. You are expected to provide the descriptive statistics for the Boston dataset. You can make use of
#the following piece of codes to have a gentle start for this question:
#http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html
#http://www.neural.cz/dataset-exploration-boston-house-pricing.html
import pandas as pd

from sklearn.datasets import load_boston
dataset = load_boston()
#print(dataset['data'])
print(dataset['feature_names'])
    #['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B' 'LSTAT']
#print(dataset.DESCR)
#dataset['target']
print("Properties: ", dataset.data.shape)
print("Desc of each column: ", dataset.DESCR)
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
#df['target'] = dataset.target
print(df)

print("Description of columns: ")
df.describe()
#Describe yerine tek tek bu fonksiyonları da çalıştırabilirsin.
    #df.count()
    #df.min()
    #df.max()
    #df.median()
print("Nullity Check:\n",pd.isnull(df).any())
print("Sum of null values for each column:\n",pd.isnull(df).sum()) #node null value at all !
print("Correlation:\n ", df.corr(method='pearson'))
print("Max TAX per each ZN\n",df.groupby(["ZN"],sort=False)['TAX'].max())
print("Min TAX per each ZN\n",df.groupby(["ZN"],sort=False)['TAX'].min())

#Q5. You are expected to provide a series of plots that you find logical for describing the Boston dataset
#as we did in the class for the Iris dataset. You do not have to provide all the different kind of figures
#but try to be precise while trying to illustrate the dataset. You can make use of the following link for the seaborn library:
#http://seaborn.pydata.org/examples/
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import scipy
warnings.filterwarnings('ignore')

from sklearn.datasets import load_boston
dataset = load_boston()
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
 #['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B' 'LSTAT']
#1
plt.hist(df['ZN'])
plt.hist(df['AGE'],bins=20)
plt.hist(df['CRIM'])
plt.hist(df['INDUS'],bins=20)
plt.hist(df['RAD'],bins=20)

#2.1
sns.distplot(df['ZN'])

#2.2.
#sns.kdeplot(df['ZN'], df['CRIM'], shade=True)
sns.jointplot(df['ZN'], df['CRIM'], kind='kde')

#3
cat_attr = df['RAD']
h = cat_attr.value_counts()
values, counts = h.index, h
plt.bar(values, counts)
plt.bar(*list(zip(*cat_attr.value_counts().items())))


#4
plt.scatter(df['RAD'], df['LSTAT'])

#5
plt.scatter(df['AGE'], df['RM'])
