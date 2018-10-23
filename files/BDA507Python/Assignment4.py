#Q1. Please prepare a number array of 3 rows and 7 columns. The content should be the numbers from
#9 to 100s with the incrementation value of 7. Then you should make a clone of this array with a
#different name. Then this copy array’s values should be changed to remainder of 8. For instance, the
#initial array held the values: 9, 16, 23, 30, …. After being copied to a new array the values should be
#adjusted as such: 1, 0, 7, 6 …. (these are the remainder when the initial values are divided by 8).
print("**********Task 1******************")
import numpy as np
x=np.arange(9,101,7)
#print(x)
x_reshape=x.reshape(2,7)
#print(x_reshape)
y=x_reshape
#print(y)
i=0
while i<2:
    j=0
    while j<7:
        y[i,j]=(y[i,j]%8)
        j+=1
    i+=1
print(y)

#Prepare a numpy array of 7 x 7 matrix with all its content having the value 9. Then please change
#the border content to 8 and the inside content to 7 separately. Please try to do it with the short cuts
#that we have done in the class (this is Part-A) as well as using 2 for loops (this is Part-B).
print("**********Task 2*********yapamadımmmmm*********")
x = x = np.full((7, 7), 9, dtype=np.uint)
#print(x)
print("*********")
#x[1:6:1,1:6:1]=8
#print(x)
x[1:-1,1:-1] = 7
print(x)
print("*********")
#x[0:7:6,0:7:6]=8
##???
#print(x)

for i in (0,6):
    for j in range(0,7):
        x[i,j]=8
print(x)

#Q3. Please prepare the checkerboard pattern with using for loops as opposed to the one that we have
#done during our last class. The code for checkerboard pattern that we have done during class is as
print("**********Task 3******************")
x = np.zeros((8,8),dtype=int)
i=1
while i<8:
    j=0
    while j<8:
        x[i,j]=1
        j+=2
    i+=2
i=0
while i<8:
    j=1
    while j<8:
        x[i,j]=1
        j+=2
    i+=2
print(x)

#Q4. For Titanic dataset, please work on the “test.csv” file this time and do the following tasks (The file
#is also in the relevant folder for Week-4):
#(a) Get the dataset via pandas library,
# (b) display the dimensions (rows and columns),
# (c) show the first 10 lines,
# (d) show the descriptive statistics both for numeric and categorical data,
# (e)change the survived column to categorical data (as “yes” and “no” values),
# (f) display the number of passengers that have used A-B-C-D-E-F cabins,
# (g) display the number of passengers whose age is greater than 40 and male (then count the females that are greater 40 years of age),
# (h) count the number of missing values for Age column.
print("**********Task 4******************")
import numpy as np
import pandas as pd
titanic = pd.read_csv("train.csv")  #(a)
print("Dimensions: ",titanic.shape) #(b)
#print("",titanic.dtypes) # data type
print("First 10 Lines: ", titanic.head(10)) #(c)
print("Statistics about data: ", titanic.describe()) #(d)
#(e)
new_survived = pd.Categorical(titanic["Survived"],ordered=True)
#print(new_survived)
new_survived = new_survived.rename_categories(["Died","Survived"])
titanic["Survived"]=new_survived
print("First 10 Lineswith new Survived col: ", titanic.head(10))
titanic.hist(column="Survived",figsize=(9,6),bins=20)
#(f)
char_cabin = titanic["Cabin"].astype(str) # Convert data to str
new_Cabin = np.array([cabin[0] for cabin in char_cabin]) # Take first letter
notnan_ind=np.where(new_Cabin!='n')
new_Cabin=new_Cabin[notnan_ind]
new_Cabin=pd.Categorical(new_Cabin)
print("#Passenger due to Cabin Cat: ",new_Cabin .describe())
#(g)display the number of passengers whose age is greater than 40 and male (then count the females that are greater 40 years of age),
oldermale= np.where((titanic["Age"]>=40) & (titanic["Sex"]=="male"))
print("Older than 40 male: ",titanic.loc[oldermale])
olderfemale= np.where((titanic["Age"]>=40) & (titanic["Sex"]=="female"))
print("# older than 40 female: ", len(olderfemale[0])),
#(h)
missing = np.where(titanic["Age"].isnull() == True)
print("Missing value count on Age Colomn: ", len(missing[0]))

#Q5. Please plot the histograms for each of the columns in the dataset. You can decide on the details
#(such as figure sizes and bins) of the figures. The code that we have gone over in the class includes
#the relevant function you need to run.
print("**********Task 5******************")
import pandas as pd
import matplotlib.pyplot as plt
titanic = pd.read_csv("train.csv")
print(titanic.dtypes)
titanic.hist(column="PassengerId",figsize=(9,6),bins=20)
titanic.hist(column="Pclass",figsize=(9,6),bins=20)
titanic["Survived"].plot.hist()
titanic["Sex"].plot.line()
titanic.hist(column="SibSp",figsize=(9,6),bins=20)
titanic.hist(column="Age",figsize=(9,6),bins=20)
titanic.hist(column="Parch",figsize=(9,6),bins=20)
titanic.hist(column="Ticket",figsize=(9,6),bins=20)
titanic.hist(column="Fare",figsize=(9,6),bins=20)
titanic.hist(column="Cabin",figsize=(9,6),bins=20)
titanic.hist(column="Embarked",figsize=(9,6),bins=20)
titanic.hist(column="Name",figsize=(9,6),bins=20)


##titanic.hist(by=titanic["Sex"]) güzel.
#
#titanic.hist(column='Age',    # Column to plot
#                   figsize=(9,6),   # Plot size
#                   bins=20)         # Number of histogram bins
##  array([[<matplotlib.axes._subplots.AxesSubplot object at 0x00000000086F59B0>]], dtype=object)
#
#titanic["Survived"].plot.bar()
#titanic["Survived"].plot.hist()
#







