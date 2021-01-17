# Intermediate Machine Learning
Learn to handle missing values, non-numeric values, data leakage and more.

## Missing Values
Approaches to dealing with missing values. Then you'll compare the effectiveness of these approaches on a real-world dataset.

If you try to build a model using data with **missing values in libraries (including scikit-learn) give an error**. So you'll need to choose one of the strategies below.

### Three Approaches to deal with missing values
  1. Simple Option: Drop Columns with Missing Values
  2. Standard approach: Imputation (A Better Option)
  3. An Extension To Imputation

#### Simple Option: Drop Columns with Missing Values
![drop entire column](drop-column.png)
The model **loses access to a lot of (potentially useful!) information** with this approach. As an extreme example, consider a dataset with 10,000 rows, where one important column is missing a single entry. This approach would drop the column entirely!

#### Standard approach: Imputation (A Better Option)
![input fills in the missing values](imputation.png)
Imputation fills in the missing values with some number. For instance, we can fill in the mean value along each column. **The imputed value won't be exactly right in most cases, but it usually leads to more accurate models** than you would get from dropping the column entirely.

#### An Extension To Imputation
The model would make **better predictions by considering which values were originally missing**.
In some cases, this will meaningfully improve results. In other cases, it doesn't help at all.
![Considering which values were originally missing](inputation-extension.png)

#### Example
In the example, we will work with the Melbourne Housing dataset. Our model will use information such as the number of rooms and land size to predict home price.
~~~
import pandas as pd
from sklearn.model_selection import train_test_split
from sklear.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# load file
fpath = "mdrent.csv"
fhand = pd.read_csv(fpath)

# target & predictor variable
y = fhand.Price
# to keep things simple, we use only numerical predictors
columns = fhand.drop(['Price'],axis=1)
x = columns.select_dtypes(exclude=['object'])

# training and validation subsets
train_x, vtion_x, train_y, vtion_y = train_test_split(x, y, train_size=0.8, test_size=0.2, random_state=0)

# function for comparing different approaches
def score_dataset(train_x, vtion_x, train_y, vtion_y):
  model = RandomForestRegressor(n_estimators=10, random_state=0)
  model.fit(train_x, train_y)
  preds = model.predict(vtion_x)
  return mean_absolute_error(vtion_y, preds)

# Get names of columns with missing values
cols_with_missing = [col for col in train_x.columns if train_x[col].isnull().any()]

# Approach 1 (Drop Columns with Missing Values)
# Drop columns in training and validation data
reduced_train_x = train_x.drop(cols_with_missing, axis=1)
reduced_vtion_x = vtion_x.drop(cols_with_missing, axis=1)

print("MAE from Approach 1 (Drop Columns with Missing Values):")
print(score_dataset(reduced_train_x, reduced_vtion_x, train_y, vtion_y))
~~~


# Categorical Variables
The data is categorical because responses fall into a fixed set of categories.

**You will get an error if you try to plug these variables into most machine learning models in Python without preprocessing them first**.

We'll compare three approaches that you can use to prepare your categorical data.

### Drop Categorical Variables
**The easiest approach** to dealing with categorical variables is to simply remove them from the dataset. This approach **will only work well if the columns did not contain useful information**.

### Label Encoding
Label encoding assigns each category to a unique integer.
![label encoding](label-encoding.png)
This approach assumes an ordering of the categories: "Never" (0) < "Rarely" (1) < "Most days" (2) < "Every day" (3).

This assumption makes sense in this example, because there is an indisputable ranking to the categories. Not all categorical variables have a clear ordering in the values, but we refer to those that do as ordinal variables. For tree-based models (like decision trees and random forests), you can expect **label encoding to work well with ordinal variables**.


### One-Hot Encoding
One-hot encoding creates new columns indicating the presence (or absence) of each possible value in the original data.
![One-Hot encoding](one-hot_encoding.png)
In the original dataset, "Color" is a categorical variable with three categories: "Red", "Yellow", and "Green". The corresponding one-hot encoding contains one column for each possible value. If the value was "Yellow", we put a 1 in the "Yellow" column, if the value was "Red", we put a 1 in the "Red" column and so on.

In contrast to label encoding, one-hot encoding does not assume an ordering of the categories. Thus, you can expect this approach to work particularly well if there is no clear ordering in the categorical data (e.g., "Red" is neither more nor less than "Yellow"). We refer to categorical variables without an intrinsic ranking as **nominal variables**. One-hot encoding generally does not perform well if the categorical variable takes on a large number of values (i.e., you generally **won't use it for variables taking more than 15 different values**).

### Example
Next, we obtain a list of all of the categorical variables in the training data.

We do this by checking the data type (or dtype) of each column. The object dtype indicates a column has text (there are other things it could theoretically be, but that's unimportant for our purposes). For this dataset, the columns with text indicate categorical variables.
~~~
s = (train_x.dtypes == 'object')
object_cols = list(s[s].index)
print("Categorical variables:")
print(object_cols)
~~~

























































































































#
