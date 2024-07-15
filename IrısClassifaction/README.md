We will use a small dataset from Fisher 1936.

One of the earliest known dataset that is used for classification.

Tabular data set of 4 features and 150 instances

can be installed from link  
https://archive.ics.uci.edu/dataset/53/iris

or directly be imported with

```
pip install ucimlrepo

```

from ucimlrepo import fetch_ucirepo

````
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

````
