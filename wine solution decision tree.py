import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
 
df = pd.read_csv('winequality.csv') 
  
  
  
X=df.iloc[:,[8,9,11,12]].values  #ph,alcohol,quality
print(X[0:5])

#print(np.isnan(X))  # will give you true if you have missing values.


Y=df.iloc[:,0].values
#print(Y.head())


from sklearn.preprocessing import LabelEncoder
# for y
y_labelencoder = LabelEncoder ()
Y = y_labelencoder.fit_transform (Y)
print (Y)



from sklearn.preprocessing import Imputer
missingValueImputer = Imputer (missing_values = 'NaN', strategy = 'mean', 
                               axis = 0)  
missingValueImputer = missingValueImputer.fit (X)

X = missingValueImputer.transform(X)



from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split (X, Y, test_size = .20, 
                                                     random_state = 0)

													 
#print("splitting done")
print(X_train.shape)		
#print(y_train)	
print(X_test.shape)	
print("Training dataset")
print(X_train)





from mlxtend.feature_selection import SequentialFeatureSelector as sfs

from sklearn.ensemble import RandomForestClassifier
RFclassifier = RandomForestClassifier(n_estimators=10, random_state=42, max_depth=4, criterion = 'entropy')

# Build step forward feature selection
sfs1 = sfs(RFclassifier,
           k_features=3,
           forward=True,
           scoring='accuracy',
           cv=5)


sfs1 = sfs1.fit(X_train, y_train)
print("next step")

feat_cols = list(sfs1.k_feature_idx_)
print(feat_cols)

RFclassifier = RandomForestClassifier(n_estimators=10, random_state=42, max_depth=4, criterion = 'entropy')
RFclassifier.fit(X_train[:,feat_cols],y_train)



prediction = RFclassifier.predict(X_test[:,feat_cols])


print(prediction)

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, prediction))

from sklearn.tree import export_graphviz

# produce decision tree graph
dot_data = export_graphviz(
        RFclassifier,
		max_depth=3,
		#feature_names=X,	
		#class_names='type',
        rounded=True,
        filled=True
    )

	
from pydotplus import graph_from_dot_data
graph = graph_from_dot_data(dot_data)
graph.write_png('dtreg.png')


import matplotlib.image as mpimg
plt.imshow(mpimg.imread('dtreg.png'))
plt.show()

"""
from sklearn.tree import export_graphviz
# Export as dot file
export_graphviz(RFclassifier, out_file='tree.dot', 
                #feature_names = iris.feature_names,
                #class_names = iris.target_names,
                rounded = True, proportion = False, 
                precision = 2, filled = True)

# Convert to png using system command (requires Graphviz)
from subprocess import call
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])

# Display in jupyter notebook
from IPython.display import Image
Image(filename = 'tree.png')



"""


