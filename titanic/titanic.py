# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 20:13:27 2018

@author: user
"""

#C:\Users\user\AppData\Roaming\SPB_Data

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('titanic/train.csv')
test = pd.read_csv('titanic/test.csv')

train['Embarked'].value_counts()
#train.head(20)
train.head()
test.head()

train.shape
test.shape

train.info()
test.info()

def bar_chart(feature):
    survived = train[train['Survived']==1][feature].value_counts()
    dead = train[train['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index = ['Survived','Dead']
    df.plot(kind='bar',stacked=True, figsize=(10,5))

survived = train[train['Survived']==1]['Pclass'].value_counts()
survived

def bar(feature):
    survived = train[train['Survived']==1][feature].value_counts()
    dead = train[train['Survived']==0][feature].value_counts()
    rate = survived / (survived + dead)
    rate.plot(kind = 'bar', figsize=(10,5))

bar('Pclass')
bar('Sex')
bar('Age')
bar('Embarked')
bar('SibSp')
bar('Parch')


bar_chart('Embarked')

train[train['Parch'] == 5]
train[train['Pclass'] == 1][train['Fare'] > 100]




train_test = [train,test]
type(train)
type(train_test)
train_test[2]
for x in train :
    x['Cabin'] = x['Cabin'].str[:1]

train['Cabin'] = train['Cabin'].str[:1]
train['Cabin']

Pclass1 = train[train['Pclass'] == 1]['Cabin'].value_counts()
Pclass1.plot(kind = 'bar', stacked = True)
train[train['Cabin']=='T']
Pclass2 = train[train['Pclass'] == 2]['Cabin'].value_counts()
Pclass2.plot(kind = 'bar', stacked = True)
Pclass3 = train[train['Pclass'] == 3]['Cabin'].value_counts()
Pclass3.plot(kind = 'bar', stacked = True)
Pclass1
Pclass2
Pclass3
train.info()

bar('Cabin')

for dataset in train:
    dataset['Cabin'] = dataset['Cabin'].str[:1]

train['FamilySize'] = train['SibSp'] + train['Parch'] + 1

bar('FamilySize')

train[train['FamilySize'] == 7][['Ticket','Age','Sex','SibSp','Parch','Survived']]
train[train['FamilySize'] == 4][['Ticket','Age','Sex','SibSp','Parch','Survived']]

train[train['FamilySize'] == 3][['Ticket','Age','Sex','SibSp','Parch','Survived']]

train[train['FamilySize'] == 1].info()
train[train['FamilySize'] == 2].info()
train[train['FamilySize'] == 3].info()
train[train['FamilySize'] == 4].info()

train.info()
train[train['Embarked'].isnull() == True]

train.loc[829:830]['Embarked'] = 'C'

train.loc[62:63]['Embarked'] = 'S'
train.loc[62]

train[train['Embarked'] == 'Q'][train['Pclass'] == 1][['Ticket','Cabin']]
train[train['Embarked'] == 'S'][train['Pclass'] == 1][train['Cabin']=='B'][['Ticket','Cabin']]
train[train['Embarked'] == 'C'][train['Pclass'] == 1][train['Cabin']=='B'][['Ticket','Cabin']]











train.head()
train['Title'] = train['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
train['Title'].value_counts()

bar('Title')


title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, 
                 "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,
                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }

train['Title'] = train['Title'].map(title_mapping)
train[train['Title'] == 1][train['Survived'] == 0]

train.drop('Name', axis=1, inplace=True)



sex_mapping = {"male": 0, "female": 1}

train['Sex'] = train['Sex'].map(sex_mapping)


train.info()
train["Age"].fillna(train.groupby("Title")["Age"].transform("median"), inplace=True)



embarked_mapping = {"S": 0, "C": 1, "Q": 2}
train['Embarked'] = train['Embarked'].map(embarked_mapping)

#bar('Embarked')
#train[train['Fare'] > 80]['Pclass']
#train[train['Fare'] > 50][train['Pclass'] != 1][['Fare','Pclass','Embarked']]
#train[train['Fare'] < 10][train['Pclass'] != 3][['Fare','Pclass','Embarked','Age']]
#
#
#train[['Fare','Pclass']]


# 1등석만 구분
#train.loc[train['Fare'] > 73.5, 'Fare'] = 1
#train.loc[train['Fare'] != 1, 'Fare'] = 2

train.drop('Cabin', axis=1, inplace=True)
train.info()
train.drop('Ticket', axis=1, inplace=True)
train.info()






from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score


#knn

k_fold = KFold(n_splits=5, shuffle=True, random_state=0)

tr = train.drop('Survived', axis=1)
tt = train['Survived']

X_train, X_test, y_train, y_test = train_test_split(tr,tt, random_state = 0)

a = []
b = []

for i in range(1,15):
    clf = KNeighborsClassifier(n_neighbors = i)
    clf.fit(X_train, y_train)
    print(clf.score(X_train, y_train))    
    print(clf.score(X_test, y_test))
    print('-----------------------------')
    


#결정트리

tree = DecisionTreeClassifier(max_depth=6)
tree.fit(X_train, y_train)
tree.score(X_train, y_train)
tree.score(X_test, y_test)

for i in range(10):
    print(i)

train.info()


a = ['PassengerId','Pclass', 'Sex', 'Age', 'SibSp','Parch','Fare','Embarked','FamilySize','Title']
for i in range(10):
    print(a[i])
    print(tree.feature_importances_[i])
    
bar('Title')    


#랜덤포레스트


forest = RandomForestClassifier(n_estimators=15)
forest.fit(X_train, y_train)
forest.score(X_train, y_train)
forest.score(X_test, y_test)

for i in range(10,30):
    forest = RandomForestClassifier(n_estimators=i)
    forest.fit(X_train, y_train)
    print(i)
    print(forest.score(X_train, y_train))
    print(forest.score(X_test, y_test))

for i in range(10) :
    print(a[i])
    print(forest.feature_importances_[i])



#SVM
    
    
svm = SVC(kernel='rbf', C=10, gamma=10).fit(X_train, y_train)
svm.score(X_train, y_train)
svm.score(X_test, y_test)

for x in [0.1, 1, 1000]:
    for y in [0.1, 1, 10]:
        svm = SVC(kernel='rbf', C=x, gamma=y).fit(X_train, y_train)
        print(x,y)
        print(svm.score(X_train, y_train))
        print(svm.score(X_test, y_test))





















