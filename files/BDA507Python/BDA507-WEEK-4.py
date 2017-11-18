

print(" FEEDBACK")
print("https://www.udacity.com/course/intro-to-data-science--ud359")

# Write a Python program to convert a list of numeric value into a one-dimensional NumPy array.
import numpy as np
l = [12.23, 13.32, 100, 36.32]
print("Original List:",l)
a = np.array(l)
print("One-dimensional numpy array: ",a)
# Original List: [12.23, 13.32, 100, 36.32]
# One-dimensional numpy array:  [  12.23   13.32  100.     36.32]


# Write a Python program to create a null vector of size 10 and update sixth value to 11.
import numpy as np
x = np.zeros(10)
print(x)
print("Update sixth value to 11")
x[6] = 11
print(x)
# [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
# Update sixth value to 11
# [  0.   0.   0.   0.   0.   0.  11.   0.   0.   0.]


# Write a Python program to reverse an array (first element becomes last).
import numpy as np
import numpy as np
x = np.arange(12, 38)
print("Original array:")
print(x)
print("Reverse array:")
x = x[::-1]
print(x)
# Original array:
# [12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37]
# Reverse array:
# [37 36 35 34 33 32 31 30 29 28 27 26 25 24 23 22 21 20 19 18 17 16 15 14 13 12]


# Write a Python program to create a array with values ranging from 12 to 38.
import numpy as np
x = np.arange(12, 38)
print(x)
# [12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37]


# Write a Python program to create a 3x3 matrix with values ranging from 2 to 10.
import numpy as np
x =  np.arange(2, 11).reshape(3,3)
print(x)
#[[ 2  3  4]
# [ 5  6  7]
# [ 8  9 10]]


# Write a Python program to an array converted to a float type.
import numpy as np
import numpy as np
a = [1, 2, 3, 4]
print("Original array")
print(a)
x = np.asfarray(a)
print("Array converted to a float type:")
print(x)
# Original array
# [1, 2, 3, 4]
# Array converted to a float type:
# [ 1.  2.  3.  4.]


#  Write a Python program to create a 2d array with 1 on the border and 0 inside.
import numpy as np
x = np.ones((5,5))
print("Original array:")
print(x)
print("1 on the border and 0 inside in the array")
x[1:-1,1:-1] = 0  ## -1 sonindex olaarak düşün. ortadaki sayıların indexini
print(x)
"""Original array:
[[ 1.  1.  1.  1.  1.]
 [ 1.  1.  1.  1.  1.]
 [ 1.  1.  1.  1.  1.]
 [ 1.  1.  1.  1.  1.]
 [ 1.  1.  1.  1.  1.]]
1 on the border and 0 inside in the array
[[ 1.  1.  1.  1.  1.]
 [ 1.  0.  0.  0.  1.]
 [ 1.  0.  0.  0.  1.]
 [ 1.  0.  0.  0.  1.]
 [ 1.  1.  1.  1.  1.]]"""


# Write a Python program to create a 8x8 matrix and fill it with a checkerboard pattern.
import numpy as np
x = np.ones((3,3))
print("Checkerboard pattern:")
x = np.zeros((8,8),dtype=int)
print(x[1::2,::2])
x[1::2,::2] = 1 ## 1::2  = 1 den sona kadar (1::) 2 şer 2şer arttır indexi ve 1 e eşitle.   ::2
x[::2,1::2] = 1
print(x)
"""Checkerboard pattern:
[[0 1 0 1 0 1 0 1]
 [1 0 1 0 1 0 1 0]
 [0 1 0 1 0 1 0 1]
 [1 0 1 0 1 0 1 0]
 [0 1 0 1 0 1 0 1]
 [1 0 1 0 1 0 1 0]
 [0 1 0 1 0 1 0 1]
 [1 0 1 0 1 0 1 0]]"""


# Write a Python program to find the set difference of two arrays.
# The set difference will return the sorted, unique values in array1 that are not in array2.
import numpy as np
array1 = np.array([0, 10, 20, 40, 60, 80])
print("Array1: ",array1)
array2 = [10, 30, 40, 50, 70]
print("Array2: ",array2)
print("Unique values in array1 that are not in array2:")
print(np.setdiff1d(array1, array2))
# Array1:  [ 0 10 20 40 60 80]
# Array2:  [10, 30, 40, 50, 70]
# Unique values in array1 that are not in array2:
# [ 0 20 60 80]


# Write a Python program to save a NumPy array to a text file.
import numpy as np
a = np.arange(1.0, 2.0, 0.2)
print(a)
np.asfarray(a)
np.savetxt('file.txt', a, delimiter='-')

# Write a Python program to create a new array of 3*5, filled with 2.
import numpy as np
#using no.full
x = np.full((3, 5), 2, dtype=np.uint)
print(x)
#using no.ones
y = np.ones([3, 5], dtype=np.uint) *2
print(y)
""""[[2 2 2 2 2]
 [2 2 2 2 2]
 [2 2 2 2 2]]
[[2 2 2 2 2]
 [2 2 2 2 2]
 [2 2 2 2 2]]"""


#  Write a Python program to Create a 1-D array of 30 evenly spaced elements between 2.5. and 6.5, inclusive.
import numpy as np
x = np.linspace(2.5, 6.5, 30)
print(x)
"""[ 2.5         2.63793103  2.77586207  2.9137931   3.05172414  3.1896551 7
  3.32758621  3.46551724  3.60344828  3.74137931  3.87931034  4.0172413 8
  4.15517241  4.29310345  4.43103448  4.56896552  4.70689655  4.8448275 9
  4.98275862  5.12068966  5.25862069  5.39655172  5.53448276  5.6724137 9
  5.81034483  5.94827586  6.0862069   6.22413793  6.36206897  6.5]"""


# Write a Python program to collapse a 3-D array into one dimension array.
import numpy as np
x = np.eye(3)
print("3-D array:")
print(x)
f = np.ravel(x, order='F')
print("One dimension array:")
print(f)
"""3-D array:
[[ 1.  0.  0.]
 [ 0.  1.  0.]
 [ 0.  0.  1.]]
One dimension array:
[ 1.  0.  0.  0.  1.  0.  0.  0.  1.]"""



#####################################################################################
#####################################################################################
################################# TITANIC DATASET ###################################
#####################################################################################
#####################################################################################
# http://hamelg.blogspot.com.tr/2015/11/python-for-data-analysis-part-14.html

# %matplotlib inline
import numpy as np
import pandas as pd
import os
# os.chdir('C:\\Users\\Greg\\Desktop\\Kaggle\\titanic') # Set working directory
titanic_train = pd.read_csv("train.csv")
titanic_train.shape           # Check dimensions
#print(titanic_train.shape)
# (889, 12)
titanic_train.dtypes
#print(titanic_train.dtypes)
"""PassengerId      int64
Survived         int64
Pclass           int64
Name            object
Sex             object
Age            float64
SibSp            int64
Parch            int64
Ticket          object
Fare           float64
Cabin           object
Embarked        object
dtype: object"""

print(titanic_train.head(5))  # Check the first 5 rows
"""PassengerId  Survived  Pclass  \
0            1         0       3
1            2         1       1
2            3         1       3
3            4         1       1
4            5         0       3"""

print( titanic_train.describe())
"""    PassengerId    Survived      Pclass         Age       SibSp  \
count   889.000000  889.000000  889.000000  712.000000  889.000000
mean    446.000000    0.382452    2.311586   29.642093    0.524184
std     256.998173    0.486260    0.834700   14.492933    1.103705
min       1.000000    0.000000    1.000000    0.420000    0.000000
25%     224.000000    0.000000    2.000000   20.000000    0.000000
50%     446.000000    0.000000    3.000000   28.000000    0.000000
75%     668.000000    1.000000    3.000000   38.000000    1.000000
max     891.000000    1.000000    3.000000   80.000000    8.000000"""


categorical = titanic_train.dtypes[titanic_train.dtypes == "object"].index
print(categorical)
#  Index(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], dtype='object')

titanic_train[categorical].describe()
print(titanic_train[categorical].describe())

print(titanic_train["Survived"])

##Task 4: 1-0 to Yes No
#titanic_train.Survived=titanic_train.Survived.astype(str)

survived_index=np.where(titanic_train["Survived"]==1)  ## where cond koyduk
titanic_train["Survived"].loc[survived_index]="Yes"
dead_index=np.where(titanic_train["Survived"]==0)
titanic_train["Survived"].loc[dead_index]="No"
print(titanic_train["Survived"])



# VARIABLE DESCRIPTIONS:
# survival        Survival
#                 (0 = No; 1 = Yes)
# pclass          Passenger Class
#                 (1 = 1st; 2 = 2nd; 3 = 3rd)
# name            Name
# sex             Sex
# age             Age
# sibsp           Number of Siblings/Spouses Aboard
# parch           Number of Parents/Children Aboard
# ticket          Ticket Number
# fare            Passenger Fare
# cabin           Cabin
# embarked        Port of Embarkation
#                 (C = Cherbourg; Q = Queenstown; S = Southampton)


del titanic_train["PassengerId"]     # Remove PassengerId

sorted(titanic_train["Name"])[0:15]   # Check the first 15 sorted names
"""['Abbing, Mr. Anthony',
 'Abbott, Mr. Rossmore Edward',
 'Abbott, Mrs. Stanton (Rosa Hunt)',
 'Abelson, Mr. Samuel',
 'Abelson, Mrs. Samuel (Hannah Wizosky)',
 'Adahl, Mr. Mauritz Nils Martin',
 'Adams, Mr. John',
 'Ahlin, Mrs. Johan (Johanna Persdotter Larsson)',
 'Aks, Mrs. Sam (Leah Rosen)',
 'Albimona, Mr. Nassef Cassem',
 'Alexander, Mr. William',
 'Alhomaki, Mr. Ilmari Rudolf',
 'Ali, Mr. Ahmed',
 'Ali, Mr. William',
 'Allen, Miss. Elisabeth Walton']"""

titanic_train["Name"].describe()
"""count                           889
unique                          889
top       Shutes, Miss. Elizabeth W
freq                              1
Name: Name, dtype: object"""


titanic_train["Name"].describe()
"""count                           889
unique                          889
top       Shutes, Miss. Elizabeth W
freq                              1
Name: Name, dtype: object"""



titanic_train["Ticket"][0:15]       # Check the first 15 tickets
"""0            A/5 21171
1             PC 17599
2     STON/O2. 3101282
3               113803
4               373450
5               330877
6                17463
7               349909
8               347742
9               237736
10             PP 9549
11              113783
12           A/5. 2151
13              347082
14              350406
Name: Ticket, dtype: object"""


titanic_train["Ticket"].describe()
"""count          889
unique         680
top       CA. 2343
freq             7
Name: Ticket, dtype: object"""


titanic_train["Cabin"][0:15]       # Check the first 15 tickets
"""0      NaN
1      C85
2      NaN
3     C123
4      NaN
5      NaN
6      E46
7      NaN
8      NaN
9      NaN
10      G6
11    C103
12     NaN
13     NaN
14     NaN
Name: Cabin, dtype: object"""


del titanic_train["Ticket"]        # Remove Ticket




titanic_train["Cabin"].describe()  # Check number of unique cabins
"""count             201
unique            145
top       C23 C25 C27
freq                4
Name: Cabin, dtype: object"""


new_survived = pd.Categorical(titanic_train["Survived"])
print(new_survived)
new_survived = new_survived.rename_categories(["Died","Survived"])
new_survived.describe()
"""counts	freqs
categories
Died	549	0.617548
Survived	340	0.382452"""



new_Pclass = pd.Categorical(titanic_train["Pclass"], ordered=True)
new_Pclass = new_Pclass.rename_categories(["Class1","Class2","Class3"])
new_Pclass.describe()
"""counts	freqs
categories
Class1	214	0.240720
Class2	184	0.206974
Class3	491	0.552306"""

titanic_train["Pclass"] = new_Pclass
titanic_train["Cabin"].unique()   # Check unique cabins
"""array([nan, 'C85', 'C123', 'E46', 'G6', 'C103', 'D56', 'A6', 'C23 C25 C27',
       'B78', 'D33', 'B30', 'C52', 'C83', 'F33', 'F G73', 'E31', 'A5',
       'D10 D12', 'D26', 'C110', 'B58 B60', 'E101', 'F E69', 'D47', 'B86',
       'F2', 'C2', 'E33', 'B19', 'A7', 'C49', 'F4', 'A32', 'B4', 'B80',
       'A31', 'D36', 'D15', 'C93', 'C78', 'D35', 'C87', 'B77', 'E67',
       'B94', 'C125', 'C99', 'C118', 'D7', 'A19', 'B49', 'D', 'C22 C26',
       'C106', 'C65', 'E36', 'C54', 'B57 B59 B63 B66', 'C7', 'E34', 'C32',
       'B18', 'C124', 'C91', 'E40', 'C128', 'D37', 'B35', 'E50', 'C82',
       'B96 B98', 'E10', 'E44', 'A34', 'C104', 'C111', 'C92', 'E38', 'D21',
       'E12', 'E63', 'A14', 'B37', 'C30', 'D20', 'B79', 'E25', 'D46',
       'B73', 'C95', 'B38', 'B39', 'B22', 'C86', 'C70', 'A16', 'C101',
       'C68', 'A10', 'E68', 'B41', 'A20', 'D19', 'D50', 'D9', 'A23', 'B50',
       'A26', 'D48', 'E58', 'C126', 'B71', 'B51 B53 B55', 'D49', 'B5',
       'B20', 'F G63', 'C62 C64', 'E24', 'C90', 'C45', 'E8', 'B101', 'D45',
       'C46', 'D30', 'E121', 'D11', 'E77', 'F38', 'B3', 'D6', 'B82 B84',
       'D17', 'A36', 'B102', 'B69', 'E49', 'C47', 'D28', 'E17', 'A24',
       'C50', 'B42', 'C148'], dtype=object)"""

char_cabin = titanic_train["Cabin"].astype(str) # Convert data to str
print(char_cabin)
new_Cabin = np.array([cabin[0] for cabin in char_cabin]) # Take first letter
print(new_Cabin)
#new_Cabin = pd.Categorical(new_Cabin)
#print(new_Cabin)
#new_Cabin .describe()


##remove from Cabin nan vlaues
nan_ind=np.where(new_Cabin!='n')
new_Cabin=new_Cabin[nan_ind]
new_Cabin=pd.Categorical(new_Cabin)
#print(new_Cabin)
new_Cabin .describe()
print(new_Cabin .describe())

##remove
index=np.where(new_Cabin!='n')
new_Cabin2=np.delete(new_Cabin,index)
print(new_Cabin2)



"""counts	freqs
categories
A	15	0.016873
B	45	0.050619
C	59	0.066367
D	33	0.037120
E	32	0.035996
F	13	0.014623
G	4	0.004499
n	688	0.773903"""

titanic_train["Cabin"] = new_Cabin
titanic_train["Age"].describe()
missing = np.where(titanic_train["Age"].isnull() == True)
"""(array([  5,  17,  19,  26,  28,  29,  31,  32,  36,  42,  45,  46,  47,
         48,  55,  63,  64,  75,  76,  81,  86,  94, 100, 106, 108, 120,
        125, 127, 139, 153, 157, 158, 165, 167, 175, 179, 180, 184, 185,
        195, 197, 200, 213, 222, 228, 234, 239, 240, 249, 255, 259, 263,
        269, 273, 276, 283, 294, 297, 299, 300, 302, 303, 305, 323, 329,
        333, 334, 346, 350, 353, 357, 358, 363, 366, 367, 374, 383, 387,
        408, 409, 410, 412, 414, 419, 424, 427, 430, 443, 450, 453, 456,
        458, 463, 465, 467, 469, 474, 480, 484, 489, 494, 496, 501, 506,
        510, 516, 521, 523, 526, 530, 532, 537, 546, 551, 556, 559, 562,
        563, 567, 572, 577, 583, 588, 592, 595, 597, 600, 601, 610, 611,
        612, 628, 632, 638, 642, 647, 649, 652, 655, 666, 668, 673, 679,
        691, 696, 708, 710, 717, 726, 731, 737, 738, 739, 759, 765, 767,
        772, 775, 777, 782, 789, 791, 792, 814, 824, 825, 827, 830, 835,
        837, 844, 847, 857, 861, 866, 876, 886], dtype=int64),)"""

len(missing[0])
# 177

titanic_train.hist(column='Age',    # Column to plot
                   figsize=(9,6),   # Plot size
                   bins=20)         # Number of histogram bins

new_age_var = np.where(titanic_train["Age"].isnull(), # Logical check
                       28,                       # Value if check is true
                       titanic_train["Age"])     # Value if check is false
titanic_train["Age"] = new_age_var
titanic_train["Age"].describe()
"""count    889.000000
mean      29.315152
std       12.984932
min        0.420000
25%       22.000000
50%       28.000000
75%       35.000000
max       80.000000
Name: Age, dtype: float64"""

titanic_train.hist(column='Age',    # Column to plot
                   figsize=(9,6),   # Plot size
                   bins=20)         # Number of histogram bins
#  array([[<matplotlib.axes._subplots.AxesSubplot object at 0x00000000086F59B0>]], dtype=object)

titanic_train["Fare"].plot(kind="box",
                           figsize=(9,9))
#  <matplotlib.axes._subplots.AxesSubplot at 0x890ba58>

index = np.where(titanic_train["Fare"] == max(titanic_train["Fare"]) )
titanic_train.loc[index]
"""Survived	Pclass	Name	Sex	Age	SibSp	Parch	Fare	Cabin	Embarked
257	1	Class1	Ward, Miss. Anna	female	35	0	0	512.3292	n	C
678	1	Class1	Cardeza, Mr. Thomas Drake Martinez	male	36	0	1	512.3292	B	C
736	1	Class1	Lesurer, Mr. Gustave J	male	35	0	0	512.3292	B	C"""



