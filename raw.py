import pandas

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree

def logger(severity, content):
  prefix = '[CONSOLE] '
  if severity == 0: 
    prefix += '[INFO] '
  if severity == 1: 
    prefix += '[WARNING] '
  if severity == 2: 
    prefix += '[SEVERE]'
  print(prefix, content)
  
data = pandas.read_csv('rdata3.csv')
logger(0, 'Data initialized!')

x = data.drop(columns=['rating'])
y = data['rating']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

model = DecisionTreeClassifier()
model.fit(x_train, y_train)
predictions = model.predict(x_test)

tree.export_graphviz(model, out_file='data.dot')

score = accuracy_score(y_test, predictions)
logger(0, 'Accuracy Score result: {}'.format(score))
