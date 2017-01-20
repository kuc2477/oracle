from sklearn import tree
from utils import *

X, y = training_set()

model = tree.DecisionTreeClassifier()
model = model.fit(X, y)
