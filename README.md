# Market AI Demo
Detailed explantion, raw code, and project resources can be found here.

The project itself with simple notes can be found here -> https://colab.research.google.com/drive/1b78skFUQ4yacJe7bEPndQgLXaXqLAnJh?usp=sharing

# Detailed Explanation
```
import pandas
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
```
These are simply the imports. Not much to explain, their documentation can be found here **https://pandas.pydata.org/docs/ - https://scikit-learn.org/stable/index.html**.

```
data = pandas.read_csv('rdata3.csv')
```
Using the pandas library, we load the CSV to read from later. We use it CSV or a spreadsheet type format because its consistant with bar graph type variables which is how our AI model will kind of work.

```
x = data.drop(columns=['rating'])
y = data['rating']
```
We take the data from the previous line and drop the column from the CSV called 'rating'. That's because the rating is what we are trying to predict based of all the previous factors. The AI will try to create a pattern in which it will find us a new 'rating' value.

```
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
```
Useful line here. Using sklearn you can split the AI process up into 2 different parts basically. A part of it that's just testing, and the other part learning. This can increate accuarcy and speed up results. The test_size variable is what percent of resources is spent testing (which is set to 0.1 or 10%). The other 90% is spent learning.

```
model = DecisionTreeClassifier()
model.fit(x_train, y_train)
predictions = model.predict(x_test)
```
The model of AI is one of the most important choices when it comes to creating a effecient and powerful AI. Honestly a decision tree isn't the best choice for my type of data, but it does the job. A decision tree is a pretty common type of algorithm and nobody does a better job of explaing it then IBM themselves (https://www.ibm.com/topics/decision-trees). The line after setting the model is just training it or "fitting" it. We are training the x and y train variables that we split up earlier. Then we use the 10% of our test data and our newly trained model to make some predictions. We save those predictions in a variable.

```
tree.export_graphviz(model, out_file='data.dot')
```
A visual example of what the model did. Something like this:
![image](https://user-images.githubusercontent.com/106291837/170409009-87b21d3d-bc26-48b4-8c44-5db1226d0f6d.png)


```
score = accuracy_score(y_test, predictions)
logger(0, 'Accuracy Score result: {}'.format(score))
```
Finally, we just simply use the sklearn library and retrieve the accuracy of it. This model sat at around 50-70% accuracy score, which if you think about it is pretty high. For customers mainly, being able to establish which customers will like your food/product on a 1/2 basis is very good odds.
