# Project Data

We will use a small dataset from Fisher 1936.

One of the earliest known dataset that is used for classification.

Tabular data set of 4 features and 150 instances

More about can be learned from here
https://archive.ics.uci.edu/dataset/53/iris

or directly be imported with

```
pip install ucimlrepo

```

from ucimlrepo import fetch_ucirepo

```
# fetch dataset
iris = fetch_ucirepo(id=53)

# data (as pandas dataframes)
X = iris.data.features
y = iris.data.targets

# metadata
print(iris.metadata)

# variable information
print(iris.variables)

```

This data set includes variables
sepal length
sepal width
petal length
petal width

and class (Iris Setosa, Iris Versicolour, or Iris Virginica)

# Requirements

We will use python with libraries :

- scipy
- numpy
- matplotlib
- pandas
- sklearn
