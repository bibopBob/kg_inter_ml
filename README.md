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























































































































#
