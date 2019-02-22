
# coding: utf-8

# In[58]:


#!pip install graphviz
import graphviz
from sklearn import tree

X = [[0, 0], [1, 1]]  ##feature
Y = [0, 1] ## label
clf = tree.DecisionTreeClassifier() ##initialize with default values
clf = clf.fit(X, Y) ##train the model
print(clf.predict([[2., 2.]])) ##predict with [2,2]
print(clf.predict_proba([[2., 2.]]))
print(clf.apply([[2., 2.]]))
print(clf.decision_path([[2., 2.]]))
print(clf.get_params())
print(clf.score([[2., 2.]],[0]))
print(clf.score([[2., 2.]],[1]))
dot_dataX = tree.export_graphviz(clf, out_file=None)
print(dot_dataX)


# In[25]:


#Task1
from sklearn.datasets import load_iris
from sklearn import tree
import graphviz
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
X = iris.data
Y = iris.target
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)

X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)

clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100, max_depth=3, min_samples_leaf=5)
clf_gini.fit(X_train, y_train)


y_pred_gini = clf_gini.predict(X_test)
print("gini",y_pred_gini)
print(accuracy_score(y_test, y_pred_gini))


clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100, max_depth=3, min_samples_leaf=5)
clf_entropy.fit(X_train, y_train)
y_pred_entropy = clf_entropy.predict(X_test)
print("entropy", y_pred_entropy)
print(accuracy_score(y_test, y_pred_entropy))






# In[ ]:





# In[74]:


#Task 2
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split  # some documents still include the cross-validation option but it no more exists in version 18.0
import pylab as plt
import graphviz
from sklearn import tree

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)
tree =DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)
print("Accuracy on training set with Decision Tree default values: {:.3f}".format(tree.score(X_train, y_train)))
print("Accuracy on test set with Decision Tree : {:.3f}".format(tree.score(X_test, y_test)))
print("*************************************************")

for i in range(1,8):
    clf = tree.DecisionTreeClassifier()
    clf=clf.fit(X_train, y_train)
    print("MAX_DEPTH:",i)
    print("Accuracy on training set: {:.3f}".format(clf.score(X_train, y_train)))
    print("Accuracy on test set: {:.3f}".format(clf.score(X_test, y_test)))
    #dot_dataX = tree.export_graphviz(clf, out_file=None) #çalısmadı
    #print(dot_dataX)  #çalısmadı

    
#numan
from sklearn.tree import export_graphviz

for i in range (1,7):
    tree = DecisionTreeClassifier(max_depth=i, random_state=0)
    tree.fit(X_train, y_train)
    print('max_depth', i)
    print("Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train)))
    print("Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))
    tree.export_graphviz(tree, out_file="tree" + str(i) + ".dot")
#numan 


# In[62]:


#dot file
#yetkin
from sklearn import tree
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
dot_dataX = tree.export_graphviz(clf, out_file=None)
print(dot_dataX)
#yetkin


# In[77]:


import numpy as np
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)
tree =DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)


def plot_feature_importances_cancer(model):
    n_features = cancer.data.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    
print("Accuracy on training set with Decision Tree default values: {:.3f}".format(tree.score(X_train, y_train)))
print("Accuracy on test set with Decision Tree : {:.3f}".format(tree.score(X_test, y_test)))
plot_feature_importances_cancer(tree) 
#Decision tree sonucunda modelin çıktıları iyiydi fakat bu graiği çizdiğimizde tek bir feature üzerinde ağırlık olduğunu gördük.
#model bazlı importance of feature değişir.


# In[75]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)
forest = RandomForestClassifier(n_estimators=100, random_state=0)
forest.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(forest.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(forest.score(X_test, y_test)))

plot_feature_importances_cancer(forest)

