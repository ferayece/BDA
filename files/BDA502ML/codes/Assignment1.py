#1. Please run the code for digits dataset that we have worked during the last class.

#Compare and discuss the outputs for the raw and scaled data as we did in the lab.
#Please assign the random state value randomly (you need to provide a random integer assignment).

import pylab as pl
from sklearn.datasets import load_digits
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from time import time
import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split  # some documents still include the cross-validation option but it no more exists in version 18.0
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from random import randint
import pylab as plt

print("**************************Question 1**************************")
##1-Scale data:
#np.random.seed(42)  # random seeding is performed
digits = load_digits()  # the whole data set with the labels and other information are extracted
data = scale(digits.data) #scale the digit data
y = digits.target # arrange the target
X = digits.data # arrange data
rndm_state=randint(1,70)  # select a random integer value.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=rndm_state) #split the scaled data into train(70%) and test(30%) data with random_State=10
gnb = GaussianNB(priors=None)  #initialize the model without priority.
fit = gnb.fit(X_train, y_train)  # fit the model with train target and train data.
predicted = fit.predict(X_test) #predict with X_test (test data,30% of scaled digits.data)
print(confusion_matrix(y_test, predicted)) ##Matrixe bak; 0 ken modelin 0 dediği 49 tane.
print("accuracy score: ",accuracy_score(y_test, predicted)) # the use of another function for calculating the accuracy (correct_predictions / all_predictions)
print("without normalization: ", accuracy_score(y_test, predicted, normalize=False))
##2-Ham data:
print("Predict with Bulk Data: ")
#digits = load_digits()  # the whole data set with the labels and other information are extracted
data = digits.data #scale the digit data
y = digits.target # arrange the target
X = digits.data # arrange data
rndm_state=randint(1,70)  # select a random integer value for random_State
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=rndm_state) #split the scaled data into train(70%) and test(30%) data with random_State=10
gnb = GaussianNB(priors=None)  #initialize the model without priority.
fit = gnb.fit(X_train, y_train)  # fit the model with train target and train data.
predicted = fit.predict(X_test) #predict with X_test (test data,30% of scaled digits.data)
print(confusion_matrix(y_test, predicted)) ##Matrixe bak; 0 ken modelin 0 dediği 49 tane.
print("accuracy score: ",accuracy_score(y_test, predicted)) # the use of another function for calculating the accuracy (correct_predictions / all_predictions)
print("without normalization: ", accuracy_score(y_test, predicted, normalize=False))

print("COMPARISION: Accuracy is higher when we scale the data. The digits: 0,1,2,3,4,9 are better predicted on scale data while 5,6,7,8 are beter predicted on raw data. So,",
"when we consider accuracy, scale data is bit better to use in model. When we compare the confusion matrices, there are more incorrect prediction raw data. In my opinion,",
"modeling with scale data is better on this dataset. But this accuracy scores can be changeble due to random_state value.")

print("**************************Question 2**************************")
#2. Please run the same code for 5 different test/train sizes such as 0.1, 0.3, 0.5 for raw dataset.
# Compare and discuss the obtained results. Which one is the best and why do you think so?
data = digits.data #scale the digit data
y = digits.target # arrange the target
X = digits.data # arrange data
for i in [0.1, 0.3, 0.5 ]:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=i, random_state=10) #split the raw data
    gnb = GaussianNB(priors=None)  #initialize the model without priority.
    fit = gnb.fit(X_train, y_train)  # fit the model with train target and train data.
    predicted = fit.predict(X_test) #predict with X_test
    print("accuracy score: ",accuracy_score(y_test, predicted)) # the use of another function for calculating the accuracy (correct_predictions / all_predictions)

print("When test size is 10%, accuracy score has the highest value. When test size is 30%, "
      "accuracy score is decreased but still acceptable but when we split the dataset like half of it as test and half of it train data,"
      "accuracy score is increased again. Reason of it can be overfitting; 50% test data size caused overfitting. "
      "So, 10% test size is the best for this dataset")



print("**************************Question 4.a**************************")
#4a.Please run the SVC (linear) code for iris dataset (that we worked during the class).
#Please perform the same algorithm for 4 different C values that you will decide. Then
#you should plot the outputs in the 2 x 2 plots (you need to get rid of the plots for label propagation but instead use SVC).

#Note:a small value for C means the margin is calculated using many or all of the observations around the separating line (more regularization);
#a large value for C means the margin is calculated on observations close to the separating line (less regularization).
#http://scikit-learn.org/stable/tutorial/statistical_inference/supervised_learning.html

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

rng = np.random.RandomState(0)
iris = datasets.load_iris() #load iris data
X = iris.data[:, :2]  ## Take the first two features. We could avoid this by using a two-dim dataset;Sepal length and Sepal width
y = iris.target  #arrange the target
#print(iris.target)

#X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=5)

C_2d_range = [1e-2,0.1,1,10] #list of different C values.
svc_model1 = (svm.SVC(kernel='linear',C=C_2d_range[0]).fit(X, y), y)  #create SVM.SVC model with first element of C value List. (c=0.01)
svc_model2 = (svm.SVC(kernel='linear',C=C_2d_range[1]).fit(X, y), y)  #create SVM.SVC model with second element of C value List. (C=1)
svc_model3 = (svm.SVC(kernel='linear',C=C_2d_range[2]).fit(X, y), y)  #create SVM.SVC model with third element of C value List.  (C=100)
svc_model4 = (svm.SVC(kernel='linear',C=C_2d_range[3]).fit(X, y), y)  #create SVM.SVC model with forth element of C value List.  (C=)


# create a mesh to plot in
  #   x: data to base x-axis meshgrid on
  #  y: data to base y-axis meshgrid on
  #  h: stepsize for meshgrid, optional
h = .02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))   # create the value set (test dataset) with a reasonable range based on features for prediction.

#print("x_min:",x_min,"y_min,",y_min,"XX:",xx,"\nYY:",yy)

color_map = {-1: (1, 1, 1), 0: (0, 0, .9), 1: (1, 0, 0), 2: (.8, .6, 0)} #color map depends the value.

# title for the plots
titles = ['C=1e-2',
          'C=0.1',
          'C=1',
          'C=10']


for i, (clf, y_train) in enumerate((svc_model1, svc_model2,svc_model3, svc_model4)):
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    plt.subplot(2, 2, i + 1)
    #print("Ravel Creation: ",i)
    #print("ravel:" ,np.c_[xx.ravel(), yy.ravel()])
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])   #ravel()=arr.reshape(-1),make prediction, Z=predicted value set for each xx-yy pair element!
    #print("Z",Z,"type:",type(Z),"dim:",Z.dim())
    #print(i,"\n", y_train)
    #print("Z:\n",Z,"Z shape:",len(Z.reshape(-1)))
    #print("Z ravel:",Z.ravel())
    #print(len(z),len(y_train))

    #print("Z: ",Z[0])
    # Put the result into a color plot
    #print("xx.shape: ", xx.shape)
    Z = Z.reshape(xx.shape)
    #print("Z_Reshape: ",Z)
    #print("Accuracy score for ",i, accuracy_score(Z[:0],y_train))
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
    plt.axis('off')
    # Plot also the training points
    colors = [color_map[y] for y in y_train]
    plt.scatter(X[:, 0], X[:, 1], c=colors, edgecolors='black')
    plt.title(titles[i])
plt.suptitle("Unlabeled points are colored white", y=0.1)
plt.show()

print("**************************Question 4.b**************************")
#4b. Please run the same SVC (linear kernel) for the iris dataset (that we worked during the class). Please perform the same algorithm for 4 different gamma values that you
#will decide. Then you should plot the outputs in the 2 x 2 plots (again you need to get rid of the plots for label propagation but instead use SVC).

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm

rng = np.random.RandomState(0)
iris = datasets.load_iris() #load iris data
X = iris.data[:, :2]  #take first two column as features.
y = iris.target  #arrange the target
#print(iris.target)
h = .02
gamma_2d_range = [1e-2, 1, 1e2,1e4] #list of different gamma values.
svc_model1 = (svm.SVC(kernel='linear',gamma=gamma_2d_range[0]).fit(X, y), y)  #create SVM.SVC model with first element of gamma value List. (gamma=0.01)
svc_model2 = (svm.SVC(kernel='linear',gamma=gamma_2d_range[1]).fit(X, y), y)  #create SVM.SVC model with second element of gamma value List. (gamma=1)
svc_model3 = (svm.SVC(kernel='linear',gamma=gamma_2d_range[2]).fit(X, y), y)  #create SVM.SVC model with third element of gamma value List.  (gamma=100)
svc_model4 = (svm.SVC(kernel='linear',gamma=gamma_2d_range[3]).fit(X, y), y)  #create SVM.SVC model with forth element of gamma value List.  (gamma=1e4)

# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

color_map = {-1: (1, 1, 1), 0: (0, 0, .9), 1: (1, 0, 0), 2: (.8, .6, 0)}

# title for the plots
titles = ['gamma=1e-2',
          'gamma=1',
          'gamma=1e2',
          'gamma=1e4']

for i, (clf, y_train) in enumerate((svc_model1, svc_model2,svc_model3, svc_model4)):
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    plt.subplot(2, 2, i + 1)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
    plt.axis('off')
    # Plot also the training points
    colors = [color_map[y] for y in y_train]
    plt.scatter(X[:, 0], X[:, 1], c=colors, edgecolors='black')
    plt.title(titles[i])
plt.suptitle("Unlabeled points are colored white", y=0.1)
plt.show()


#5. List all the outputs for accuracy values that you have found in question-4a and
#question-4b. Compare and discuss the results of the 8 models that you obtained.

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

rng = np.random.RandomState(0)
iris = datasets.load_iris() #load iris data
X = iris.data[:, :2]  ## Take the first two features. We could avoid this by using a two-dim dataset;Sepal length and Sepal width
y = iris.target  #arrange the target

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=5)

#accuracy compare due to different C values:
#-C:controles tradeoff between smooth decision boundary and classifying training points correctly.
#a large value for C means the margin is calculated on observations close to the separating line (less regularization).
gamma_2d_range = [1e-2, 1, 1e2,1e4]
acc_list=[]
for g in gamma_2d_range:
    clf=SVC(kernel="linear",gamma=g)
    clf.fit(X_train,y_train)
    pred=clf.predict(X_test)
    acc=accuracy_score(pred,y_test)
    acc_list.append(acc)
    print("Accuracy_score for gamma=",g,":",acc)

plt.plot(gamma_2d_range,acc_list,lw=1)
plt.ylabel('Accuracy Scores due to Different gamma values')
plt.show()

#1. Please run the code for digits dataset that we have worked during the last class.
#Compare and discuss the outputs for the raw and scaled data as we did in the lab.
#Please assign the random state value randomly (you need to provide a random integer assignment).

import pylab as pl
from sklearn.datasets import load_digits
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from time import time
import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split  # some documents still include the cross-validation option but it no more exists in version 18.0
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from random import randint
import pylab as plt

print("**************************Question 1**************************")
##1-Scale data:
#np.random.seed(42)  # random seeding is performed
digits = load_digits()  # the whole data set with the labels and other information are extracted
data = scale(digits.data) #scale the digit data
y = digits.target # arrange the target
X = digits.data # arrange data
rndm_state=randint(1,70)  # select a random integer value.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=rndm_state) #split the scaled data into train(70%) and test(30%) data with random_State=10
gnb = GaussianNB(priors=None)  #initialize the model without priority.
fit = gnb.fit(X_train, y_train)  # fit the model with train target and train data.
predicted = fit.predict(X_test) #predict with X_test (test data,30% of scaled digits.data)
print(confusion_matrix(y_test, predicted)) ##Matrixe bak; 0 ken modelin 0 dediği 49 tane.
print("accuracy score: ",accuracy_score(y_test, predicted)) # the use of another function for calculating the accuracy (correct_predictions / all_predictions)
print("without normalization: ", accuracy_score(y_test, predicted, normalize=False))
##2-Ham data:
print("Predict with Bulk Data: ")
#digits = load_digits()  # the whole data set with the labels and other information are extracted
data = digits.data #scale the digit data
y = digits.target # arrange the target
X = digits.data # arrange data
rndm_state=randint(1,70)  # select a random integer value for random_State
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=rndm_state) #split the scaled data into train(70%) and test(30%) data with random_State=10
gnb = GaussianNB(priors=None)  #initialize the model without priority.
fit = gnb.fit(X_train, y_train)  # fit the model with train target and train data.
predicted = fit.predict(X_test) #predict with X_test (test data,30% of scaled digits.data)
print(confusion_matrix(y_test, predicted)) ##Matrixe bak; 0 ken modelin 0 dediği 49 tane.
print("accuracy score: ",accuracy_score(y_test, predicted)) # the use of another function for calculating the accuracy (correct_predictions / all_predictions)
print("without normalization: ", accuracy_score(y_test, predicted, normalize=False))

print("COMPARISION: Accuracy is higher when we scale the data. The digits: 0,1,2,3,4,9 are better predicted on scale data while 5,6,7,8 are beter predicted on raw data. So,",
"when we consider accuracy, scale data is bit better to use in model. When we compare the confusion matrices, there are more incorrect prediction raw data. In my opinion,",
"modeling with scale data is better on this dataset. But this accuracy scores can be changeble due to random_state value.")

print("**************************Question 2**************************")
#2. Please run the same code for 5 different test/train sizes such as 0.1, 0.3, 0.5 for raw dataset.
# Compare and discuss the obtained results. Which one is the best and why do you think so?
data = digits.data #scale the digit data
y = digits.target # arrange the target
X = digits.data # arrange data
for i in [0.1, 0.3, 0.5 ]:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=i, random_state=10) #split the scaled data into train(70%) and test(30%) data with random_State=10
    gnb = GaussianNB(priors=None)  #initialize the model without priority.
    fit = gnb.fit(X_train, y_train)  # fit the model with train target and train data.
    predicted = fit.predict(X_test) #predict with X_test (test data,30% of scaled digits.data)
    print("accuracy score: ",accuracy_score(y_test, predicted)) # the use of another function for calculating the accuracy (correct_predictions / all_predictions)
print("!!!!!!!!!!!!!!!HATA:!!!!!!!!!!!MY OPINION: while test_Size is increasing, accuracy is decreasing. So; reasonable test size with reasonable accuracy is the best selection for this dataset.")
# DECISION: while test_Size is increasing, accuracy is decreasing. So; reasonable test size with reasonable accuracy is the best selection for this dataset.


print("**************************Question 3**************************")
#3. Please explain how the Gaussian Naive Bayesian model. How does Gaussian Naive
#Bayes algorithm work in the digits dataset? Assume that you are explaining it to a close
#friend of yours who has no background in data science but she is highly willing to learn
#it. Please do not forget to refer to the features, independence assumption, and labels.
print("Naive Bayes is a type of supervised learning method and it is based on Bayes Theorem. It is called 'Naive'"
"because these alghorithms assumes that all features are independent from each other. This assumption seperate the Naive Bayes algorithm"
"from other supervised learning techniques in Machine Learning. Naive Bayes has different algorithms such as Gaussian and Multinominal."
"On digit dataset, Gaussian Naive Bayes algorithm which assumes the likelihood of the features are Gaussian was impelemented."
"Digit dataset contains images of hand-written digits: 10 classes where each class refers to a digit. digits.data includes "
"1797 sample data,for each sample, it holds an array with sizez 64. (1797*64) So, digits.data has 1797 digital representations"
"of hand-writen numbers(0 to 9) which are features for Gaussian Naive Bayes. "
"Labels are numbers from 0 to 9 which are target for Gaussian Naive Bayes. So, basically Naive Bayes learn from data to make classification" 
"according to target.")

print("**************************Question 4.a**************************")
#4a.Please run the SVC (linear) code for iris dataset (that we worked during the class).
#Please perform the same algorithm for 4 different C values that you will decide. Then
#you should plot the outputs in the 2 x 2 plots (you need to get rid of the plots for label propagation but instead use SVC).

#Note:a small value for C means the margin is calculated using many or all of the observations around the separating line (more regularization);
#a large value for C means the margin is calculated on observations close to the separating line (less regularization).
#http://scikit-learn.org/stable/tutorial/statistical_inference/supervised_learning.html

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

rng = np.random.RandomState(0)
iris = datasets.load_iris() #load iris data
X = iris.data[:, :2]  ## Take the first two features. We could avoid this by using a two-dim dataset;Sepal length and Sepal width
y = iris.target  #arrange the target
#print(iris.target)

#X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=5)

C_2d_range = [1e-2,0.1,1,10] #list of different C values.
svc_model1 = (svm.SVC(kernel='linear',C=C_2d_range[0]).fit(X, y), y)  #create SVM.SVC model with first element of C value List. (c=0.01)
svc_model2 = (svm.SVC(kernel='linear',C=C_2d_range[1]).fit(X, y), y)  #create SVM.SVC model with second element of C value List. (C=1)
svc_model3 = (svm.SVC(kernel='linear',C=C_2d_range[2]).fit(X, y), y)  #create SVM.SVC model with third element of C value List.  (C=100)
svc_model4 = (svm.SVC(kernel='linear',C=C_2d_range[3]).fit(X, y), y)  #create SVM.SVC model with forth element of C value List.  (C=)


# create a mesh to plot in
  #   x: data to base x-axis meshgrid on
  #  y: data to base y-axis meshgrid on
  #  h: stepsize for meshgrid, optional
h = .02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))   # create the value set (test dataset) with a reasonable range based on features for prediction.

#print("x_min:",x_min,"y_min,",y_min,"XX:",xx,"\nYY:",yy)

color_map = {-1: (1, 1, 1), 0: (0, 0, .9), 1: (1, 0, 0), 2: (.8, .6, 0)} #color map depends the value.

# title for the plots
titles = ['C=1e-2',
          'C=0.1',
          'C=1',
          'C=10']


for i, (clf, y_train) in enumerate((svc_model1, svc_model2,svc_model3, svc_model4)):
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    plt.subplot(2, 2, i + 1)
    #print("Ravel Creation: ",i)
    #print("ravel:" ,np.c_[xx.ravel(), yy.ravel()])
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])   #ravel()=arr.reshape(-1),make prediction, Z=predicted value set for each xx-yy pair element!
    #print("Z",Z,"type:",type(Z),"dim:",Z.dim())
    #print(i,"\n", y_train)
    #print("Z:\n",Z,"Z shape:",len(Z.reshape(-1)))
    #print("Z ravel:",Z.ravel())
    #print(len(z),len(y_train))

    #print("Z: ",Z[0])
    # Put the result into a color plot
    #print("xx.shape: ", xx.shape)
    Z = Z.reshape(xx.shape)
    #print("Z_Reshape: ",Z)
    #print("Accuracy score for ",i, accuracy_score(Z[:0],y_train))
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
    plt.axis('off')
    # Plot also the training points
    colors = [color_map[y] for y in y_train]
    plt.scatter(X[:, 0], X[:, 1], c=colors, edgecolors='black')
    plt.title(titles[i])
plt.suptitle("Unlabeled points are colored white", y=0.1)
plt.show()

print("**************************Question 4.b**************************")
#4b. Please run the same SVC (linear kernel) for the iris dataset (that we worked during the class). Please perform the same algorithm for 4 different gamma values that you
#will decide. Then you should plot the outputs in the 2 x 2 plots (again you need to get rid of the plots for label propagation but instead use SVC).

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm

rng = np.random.RandomState(0)
iris = datasets.load_iris() #load iris data
X = iris.data[:, :2]  #take first two column as features.
y = iris.target  #arrange the target
#print(iris.target)
h = .02
gamma_2d_range = [1e-2, 1, 1e2,1e4] #list of different gamma values.
svc_model1 = (svm.SVC(kernel='linear',gamma=gamma_2d_range[0]).fit(X, y), y)  #create SVM.SVC model with first element of gamma value List. (gamma=0.01)
svc_model2 = (svm.SVC(kernel='linear',gamma=gamma_2d_range[1]).fit(X, y), y)  #create SVM.SVC model with second element of gamma value List. (gamma=1)
svc_model3 = (svm.SVC(kernel='linear',gamma=gamma_2d_range[2]).fit(X, y), y)  #create SVM.SVC model with third element of gamma value List.  (gamma=100)
svc_model4 = (svm.SVC(kernel='linear',gamma=gamma_2d_range[3]).fit(X, y), y)  #create SVM.SVC model with forth element of gamma value List.  (gamma=1e4)

# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

color_map = {-1: (1, 1, 1), 0: (0, 0, .9), 1: (1, 0, 0), 2: (.8, .6, 0)}

# title for the plots
titles = ['gamma=1e-2',
          'gamma=1',
          'gamma=1e2',
          'gamma=1e4']

for i, (clf, y_train) in enumerate((svc_model1, svc_model2,svc_model3, svc_model4)):
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    plt.subplot(2, 2, i + 1)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
    plt.axis('off')
    # Plot also the training points
    colors = [color_map[y] for y in y_train]
    plt.scatter(X[:, 0], X[:, 1], c=colors, edgecolors='black')
    plt.title(titles[i])
plt.suptitle("Unlabeled points are colored white", y=0.1)
plt.show()

print("**************************Question 5**************************")
#5. List all the outputs for accuracy values that you have found in question-4a and
#question-4b. Compare and discuss the results of the 8 models that you obtained.

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

rng = np.random.RandomState(0)
iris = datasets.load_iris() #load iris data
X = iris.data[:, :2]  ## Take the first two features. We could avoid this by using a two-dim dataset;Sepal length and Sepal width
y = iris.target  #arrange the target

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=5)

#accuracy compare due to different C values:
#-C:controles tradeoff between smooth decision boundary and classifying training points correctly.
#a large value for C means the margin is calculated on observations close to the separating line (less regularization).
gamma_2d_range = [0.00001,1e-2,0.1,0.5,1,10]
acc_list=[]
for c in C_2d_range:
    clf=SVC(kernel="linear",C=c)
    clf.fit(X_train,y_train)
    pred=clf.predict(X_test)
    acc=accuracy_score(pred,y_test)
    acc_list.append(acc)
    print("Accuracy_score for gamma=",c,":",acc)

plt.plot(C_2d_range,acc_list,lw=1)
plt.ylabel('Accuracy Scores due to Different C values')
plt.show()

print("when C=0.01, accuracy is low. When c=0.1; accuracy has the highest value and after this value; accuracy is firstly decreasing but after "
      "one break value, accuracy score is a constant value ; so after this break point C value does not affect the model.To sum up,"
      "C has effect on SVC with kernel=linear but some points it has little effect that cannot be recognizable without graph. ")

#accuracy compare due to different GAMMA values:
#-Gamma: defines how far the influence of a single training example reaches.
gamma_2d_range = [1e-2, 1, 1e2,1e4]
acc_list=[]
acc_list_rbf=[]
for g in gamma_2d_range:
    clf=SVC(kernel="linear",gamma=g)
    clf.fit(X_train,y_train)
    pred=clf.predict(X_test)
    acc=accuracy_score(pred,y_test)
    acc_list.append(acc)
    #print("Accuracy_score for gamma",g,":",acc)
    clf=SVC(kernel="rbf",gamma=g)
    clf.fit(X_train,y_train)
    pred=clf.predict(X_test)
    acc=accuracy_score(pred,y_test)
    acc_list_rbf.append(acc)
    print("Accuracy_score for gamma=",g,":",acc)


plt.plot(gamma_2d_range,acc_list,lw=1)
plt.ylabel('Accuracy Scores due to Different gamma values with kernel=Linear')
plt.show()

plt.plot(gamma_2d_range,acc_list_rbf,lw=1)
plt.ylabel('Accuracy Scores due to Different gamma values with kernel=rbf')
plt.show()

print("Gamma has no effect on SVC when kernel=Linear.But when gamma=1 accuracy score is 0.89. So, gamma has effect "
      "when kernel=rbf but when it increases, accuracy is decreasing after one break point that is 1 for this dataset.")

#accuracy compare due to different GAMMA values:
#-Gamma: defines how far the influence of a single training example reaches.
gamma_2d_range = [0.00001,1e-2,0.1,0.5,1,10]
acc_list=[]
for c in C_2d_range:
    clf=SVC(kernel="linear",C=c)
    clf.fit(X_train,y_train)
    pred=clf.predict(X_test)
    acc=accuracy_score(pred,y_test)
    acc_list.append(acc)
    print("Accuracy_score for gamma=",c,":",acc)

plt.plot(C_2d_range,acc_list,lw=1)
plt.ylabel('Accuracy Scores due to Different C values')
plt.show()

print("**************************Question 6**************************")
#6. Please explain the need for semi-supervised learning methods. Please try to find a
#concrete example and discuss it. (At most 100 words).
print("Semi-supervised learning is a type of supervised learning which can be used on unlabeled data for training. It is generally"
      "applied on a dataset which has a small amount of labeled data with a large amount of unlabeled data but concept is using the dataset "
      "includes some label and unlabel data. It is generally use when data is hard-to-get label. For example, semi-supervised learning can be used in"
      "image recognization. Firstly, we can label some images like 'toy','table' and 'chair'. Rest of image dataset are unlabeled. Secondly,"
      "we should try to classify the unlabeled data due to their properties. Thirdly, select an image (I dont know what should be criteria to select)"
      "from a generated class and try to predict its label according to labeled images on first step. So, now, we have new labeled image and our method"
      "is self learning actually. Repeat this steps until labeling is enough for you. :)")


print("**************************Question 7**************************")
#7. Please provide two concrete examples: one of them is more suitable for applying Naïve Bayes algorithm
# and the other for Support Vector Machine algorithm. Provide a lvery rough comparison accordingly.

