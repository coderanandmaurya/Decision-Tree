import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('dataset.csv')
print(df.head())

X=df.iloc[:,0:2].values
y=df.iloc[:,2].values

print(X)
print(y)


from sklearn.preprocessing import LabelEncoder
X_labelencoder = LabelEncoder()
X[:, 0] = X_labelencoder.fit_transform(X[:, 0])
print (X)

# for y
y_labelencoder = LabelEncoder ()
y = y_labelencoder.fit_transform (y)
print (y)

y=y.reshape(-1,1)
print(y)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
"""
criterion: It defines the function to measure the impurity. Sklearn supports “gini” criteria for Gini Index
 & “entropy” for Information Gain. By default, it takes “gini” value.
 
max_depth: The max_depth parameter denotes maximum depth of the tree.
"""
from sklearn.tree import DecisionTreeClassifier
#decision_tree = DecisionTreeClassifier(criterion = "gini", max_depth = 3,random_state = 100)
#decision_tree.fit(X_train, y_train)
decision_tree = DecisionTreeClassifier(criterion = "entropy", random_state = 100,max_depth=3)
decision_tree.fit(X_train, y_train)


predictValues =decision_tree.predict(X_test)


from sklearn import metrics
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
print("RMSE is:-")
print(np.sqrt(metrics.mean_squared_error(y_test, predictValues)))


data_feature_names = [ 'color', 'diameter']

from sklearn.tree import export_graphviz

# produce decision tree graph
dot_data = export_graphviz(
        decision_tree,
		max_depth=3,
		feature_names=data_feature_names,	
		class_names='label',
        rounded=True,
        filled=True
    )

	
from pydotplus import graph_from_dot_data
graph = graph_from_dot_data(dot_data)
graph.write_png('dtreg.png')


import matplotlib.image as mpimg
plt.imshow(mpimg.imread('dtreg.png'))
plt.show()
