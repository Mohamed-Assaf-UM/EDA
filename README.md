### 1. **Imports and Random Seed**
```python
import numpy as np
import pandas as pd

# Set the random seed for reproducibility
np.random.seed(123)
```
- **`numpy`** (`np`) is used for numerical operations like generating random numbers.
- **`pandas`** (`pd`) is used to handle data in tabular format, especially for creating and manipulating DataFrames.
- **`np.random.seed(123)`** sets a fixed seed for random number generation to ensure that the results are reproducible. This ensures that every time the code runs, the same random data is generated.

### 2. **Creating an Imbalanced Dataset**
```python
# Create a dataframe with two classes
n_samples = 1000
class_0_ratio = 0.9
n_class_0 = int(n_samples * class_0_ratio)
n_class_1 = n_samples - n_class_0
n_class_0,n_class_1
```
- **`n_samples = 1000`** defines the total number of samples.
- **`class_0_ratio = 0.9`** means that 90% of the data will belong to class 0.
- **`n_class_0 = int(n_samples * class_0_ratio)`** calculates the number of class 0 samples, which is 900 (90% of 1000).
- **`n_class_1 = n_samples - n_class_0`** calculates the number of class 1 samples, which is 100 (the remaining 10%).

This results in an imbalanced dataset with 900 samples of class 0 and only 100 samples of class 1.

### 3. **Creating Data for Each Class**
```python
## CREATE MY DATAFRAME WITH IMBALANCED DATASET
class_0 = pd.DataFrame({
    'feature_1': np.random.normal(loc=0, scale=1, size=n_class_0),
    'feature_2': np.random.normal(loc=0, scale=1, size=n_class_0),
    'target': [0] * n_class_0
})

class_1 = pd.DataFrame({
    'feature_1': np.random.normal(loc=2, scale=1, size=n_class_1),
    'feature_2': np.random.normal(loc=2, scale=1, size=n_class_1),
    'target': [1] * n_class_1
})
```
- For **Class 0 (majority)**, you are creating a DataFrame with two features (`feature_1`, `feature_2`), both normally distributed around `mean=0` and `std_dev=1`. All samples in this class have a `target` of 0.
- For **Class 1 (minority)**, the same two features are normally distributed but centered around `mean=2` to create a distinction from Class 0. All samples in this class have a `target` of 1.

### 4. **Combining the DataFrames**
```python
df = pd.concat([class_0, class_1]).reset_index(drop=True)
df.tail()
```
- **`pd.concat([class_0, class_1])`** combines the majority (Class 0) and minority (Class 1) data into one DataFrame.
- **`reset_index(drop=True)`** ensures the index is reset after concatenation to avoid duplicate index values.
- **`df.tail()`** prints the last 5 rows of the combined DataFrame.

### 5. **Checking Class Distribution**
```python
df['target'].value_counts()
```
- **`df['target'].value_counts()`** displays the count of each class in the `target` column, showing that Class 0 has 900 samples and Class 1 has 100 samples—confirming the dataset is imbalanced.

---

## Upsampling (Over-Sampling)

### 6. **Separating Majority and Minority Classes**
```python
df_minority = df[df['target'] == 1]
df_majority = df[df['target'] == 0]
```
- **`df_minority = df[df['target'] == 1]`** filters the DataFrame to get only the minority class (Class 1).
- **`df_majority = df[df['target'] == 0]`** filters the DataFrame to get the majority class (Class 0).

### 7. **Resampling (Upsampling the Minority Class)**
```python
from sklearn.utils import resample

df_minority_upsampled = resample(df_minority,
                                 replace=True, # Sample with replacement
                                 n_samples=len(df_majority), # Match majority class size
                                 random_state=42
                                )
```
- **`resample`** is a function from `sklearn.utils` that allows sampling with or without replacement.
- **`replace=True`** means we are sampling *with replacement*, meaning the same sample can appear multiple times. This is necessary because we are oversampling the minority class to match the number of majority class samples.
- **`n_samples=len(df_majority)`** ensures that the minority class is upsampled to have the same number of samples as the majority class (900 samples).
- **`random_state=42`** ensures reproducibility, similar to setting a seed.

### 8. **Combining the Upsampled Minority Class with the Majority Class**
```python
df_upsampled = pd.concat([df_majority, df_minority_upsampled])
df_upsampled['target'].value_counts()
```
- **`pd.concat([df_majority, df_minority_upsampled])`** combines the majority class (Class 0) and the upsampled minority class (Class 1) into a new DataFrame `df_upsampled`.
- **`df_upsampled['target'].value_counts()`** shows that both classes now have 900 samples each, effectively balancing the dataset through upsampling.

---

### Example:
Before upsampling, you had:
- **Class 0 (majority):** 900 samples
- **Class 1 (minority):** 100 samples

After upsampling, both classes have 900 samples. The minority class (Class 1) has been artificially increased by resampling (with replacement) from its original 100 samples.

### **What is happening in Upsampling?**
Upsampling is a technique where the minority class is sampled with replacement to match the size of the majority class. This prevents the model from being biased towards the majority class, giving equal importance to both classes.

---

In **upsampling**, we increase the minority class size by duplicating its samples. We use `replace=True` because we want to allow repeated selection of the same sample (with replacement) to reach the majority class size.

In **downsampling**, we reduce the majority class size by randomly selecting a subset of its data. Here, `replace=False` is used to ensure each selected sample is unique, avoiding duplicates (without replacement).

We use these techniques to balance the dataset and prevent model bias toward the majority class.

**SMOTE (Synthetic Minority Oversampling Technique)** helps balance an imbalanced dataset by creating synthetic data points for the minority class. Let’s break down the process using your code and simple explanations.

### Steps:
1. **Data Generation**:
   ```python
   from sklearn.datasets import make_classification
   X, y = make_classification(n_samples=1000, n_redundant=0, n_features=2, n_clusters_per_class=1,
                              weights=[0.90], random_state=12)
   ```
   - Here, we create a dataset with 1000 samples, where 90% belong to class 0 (majority) and 10% to class 1 (minority).
   - The features `f1` and `f2` represent the two input features for each sample.
  
   **Example**: Imagine you have 1000 marbles, where 900 are red (class 0) and 100 are blue (class 1). This dataset is imbalanced.

2. **Visualizing the Imbalance**:
   ```python
   import pandas as pd
   final_df = pd.concat([pd.DataFrame(X, columns=['f1', 'f2']), pd.DataFrame(y, columns=['target'])], axis=1)
   plt.scatter(final_df['f1'], final_df['f2'], c=final_df['target'])
   ```
   - We plot the data to visualize how the two classes are distributed.

   **Example**: Imagine placing 100 blue marbles among 900 red ones. You’ll see a lot of red marbles clustered together, with only a few blue scattered.

3. **Applying SMOTE**:
   ```python
   from imblearn.over_sampling import SMOTE
   oversample = SMOTE()
   X, y = oversample.fit_resample(final_df[['f1', 'f2']], final_df['target'])
   ```
   - SMOTE creates synthetic examples of the minority class by generating new data points that lie between existing minority samples.
   - In this step, we balance the dataset by adding synthetic class 1 data to make both classes equal (900 each).

   **Example**: Instead of adding more blue marbles by hand, SMOTE "creates" new blue marbles by looking at the existing ones and generating new similar marbles between them.

4. **Result**:
   ```python
   plt.scatter(oversample_df['f1'], oversample_df['f2'], c=oversample_df['target'])
   ```
   - After applying SMOTE, the data is balanced, and both class 0 and class 1 have 900 samples each.
   - The plot now shows an even distribution of both classes.

**Why Use SMOTE?**
SMOTE helps when a model is biased toward the majority class in an imbalanced dataset. Instead of simply duplicating minority samples, SMOTE creates synthetic data by "interpolating" between them, making the minority class more diverse and avoiding overfitting.
**SMOTE** identifies which class is the **minority** (the one with fewer samples) and which is the **majority**. Then, it generates synthetic samples for the minority class by creating new points between existing minority data points.

Here’s how it works:
- SMOTE doesn't just duplicate the minority samples. Instead, it picks two nearby minority samples and generates a new sample between them. This helps create a more diverse set of minority data points.
  
In simple terms:
- If you have 100 blue marbles (minority) and 900 red marbles (majority), SMOTE adds **new** blue marbles by "mixing" two nearby blue marbles to create a new one, making the blue and red marbles equal in number.
This helps prevent overfitting and improves model performance on imbalanced datasets.


### 5-number summary:
This gives a quick snapshot of how your data is spread:
1. **Minimum**: The smallest value (32 in your data).
2. **Q1 (First Quartile)**: The 25th percentile, meaning 25% of the data is below this value (54).
3. **Median**: The middle value when data is sorted (67).
4. **Q3 (Third Quartile)**: The 75th percentile, meaning 75% of the data is below this value (89).
5. **Maximum**: The largest value (99).

### Interquartile Range (IQR):
- **IQR** = Q3 - Q1 = 89 - 54 = **35**. This range measures the spread of the middle 50% of your data.
  
### Detecting Outliers:
Outliers are data points that are significantly higher or lower than most of the data. To detect them:
- **Lower fence** = Q1 - 1.5 * IQR = 54 - 1.5 * 35 = **1.5**
- **Upper fence** = Q3 + 1.5 * IQR = 89 + 1.5 * 35 = **141.5**

Any data points below **1.5** or above **141.5** are considered **outliers**. In your first dataset, there are no such values, so no outliers. 

### Example with Outliers:
In the second dataset, where you added extreme values like `-100, -200, 150, 170, 180`, these are considered **outliers** because they lie outside the lower and upper fences. The box plot visually highlights these outliers.

### Box Plot:
- The box represents the middle 50% of the data (between Q1 and Q3).
- The "whiskers" extend to the smallest and largest non-outlier values.
- **Outliers** appear as points outside the whiskers (as seen in the second plot).

In summary:
- The **5-number summary** and **IQR** help you detect **outliers**.
- **Box plots** give a visual representation of your data spread and highlight any outliers.
  Let's break down the three types of data encoding—**One Hot Encoding**, **Label Encoding**, and **Ordinal Encoding**—using the code you provided along with examples for better understanding.

### 1. **One Hot Encoding (OHE) / Nominal Encoding**

**Explanation:**
One Hot Encoding transforms categorical data into a form that can be used in machine learning models by creating a binary (0 or 1) column for each unique category in the dataset.

**Example:**
Let’s say you have a column called `color` with values: `['red', 'blue', 'green']`. OHE will create new columns for each unique value of the `color` variable:

- `Red: [1, 0, 0]`
- `Blue: [0, 1, 0]`
- `Green: [0, 0, 1]`

Each row will have a 1 for its respective category, and 0 elsewhere.

#### Code:
```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Sample dataframe
df = pd.DataFrame({
    'color': ['red', 'blue', 'green', 'green', 'red', 'blue']
})

# One hot encoding
encoder = OneHotEncoder()
encoded = encoder.fit_transform(df[['color']]).toarray()

# Convert to dataframe for easier view
encoder_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out())
print(encoder_df)
```

**Output:**
```
   color_blue  color_green  color_red
0         0.0          0.0        1.0
1         1.0          0.0        0.0
2         0.0          1.0        0.0
3         0.0          1.0        0.0
4         0.0          0.0        1.0
5         1.0          0.0        0.0
```

This is useful when the categorical variable doesn’t have an inherent order.

---

### 2. **Label Encoding**

**Explanation:**
Label Encoding assigns each unique category a different integer value. This can be useful when there is no need for binary representation. However, be cautious when using label encoding for variables without a natural order, as the algorithm might misinterpret the encoding as ordinal.

**Example:**
Consider the `color` column again. Instead of creating new columns, label encoding will assign integers to each color:

- `Red: 2`
- `Green: 1`
- `Blue: 0`

#### Code:
```python
from sklearn.preprocessing import LabelEncoder

# Label encoding
lbl_encoder = LabelEncoder()
encoded_labels = lbl_encoder.fit_transform(df['color'])
print(encoded_labels)
```

**Output:**
```
[2 0 1 1 2 0]
```

Here, `red` gets encoded as 2, `blue` as 0, and `green` as 1.

---

### 3. **Ordinal Encoding**

**Explanation:**
Ordinal Encoding is used when the categorical values have a meaningful order or rank. For example, education levels like `['High school', 'College', 'Graduate', 'Post-graduate']`. Each category is assigned a number based on its rank.

**Example:**
Consider a dataset with a `size` column having values `['small', 'medium', 'large']`. Since these values have an order (small < medium < large), ordinal encoding is suitable here.

- `small: 0`
- `medium: 1`
- `large: 2`

#### Code:
```python
from sklearn.preprocessing import OrdinalEncoder

# Sample dataframe
df = pd.DataFrame({
    'size': ['small', 'medium', 'large', 'medium', 'small', 'large']
})

# Ordinal encoding
encoder = OrdinalEncoder(categories=[['small', 'medium', 'large']])
encoded = encoder.fit_transform(df[['size']])
print(encoded)
```

**Output:**
```
[[0.]
 [1.]
 [2.]
 [1.]
 [0.]
 [2.]]
```

In this case, `small` is encoded as 0, `medium` as 1, and `large` as 2, which reflects their natural order.

---

### Key Differences:
- **One Hot Encoding** creates multiple binary columns for each category, suitable for nominal (non-ordered) categorical variables.
- **Label Encoding** assigns an integer label to each category, often used when the categorical data has no specific order, but can cause issues if misinterpreted as ordinal.
- **Ordinal Encoding** is specifically for categorical data with a clear, meaningful order, such as rankings or levels.
**Target Guided Ordinal Encoding** is a technique that encodes categorical variables based on their relationship with the target variable (usually a numeric value, like "price" or "salary"). This encoding method assigns numerical values to categories by calculating a summary statistic (like the mean or median) of the target variable for each category. This helps create a numerical relationship between the categorical and target variables, which can improve the performance of machine learning models.

Here’s a simple breakdown of how **Target Guided Ordinal Encoding** works:

### Example Code and Explanation:

```python
import pandas as pd

# create a sample dataframe with a categorical variable and a target variable
df = pd.DataFrame({
    'city': ['New York', 'London', 'Paris', 'Tokyo', 'New York', 'Paris'],
    'price': [200, 150, 300, 250, 180, 320]
})
```

#### Step 1: Grouping by the Categorical Variable
We have a dataset with two columns: `city` (categorical) and `price` (target). In this case, we want to encode the `city` variable based on the average (mean) of `price` for each city.

```python
mean_price = df.groupby('city')['price'].mean().to_dict()
mean_price
```

Here’s what happens:
- We group the `df` DataFrame by the `city` column and compute the mean of the `price` for each city.
- The result is a dictionary where each city is mapped to its corresponding mean `price`.

```python
{'London': 150.0, 'New York': 190.0, 'Paris': 310.0, 'Tokyo': 250.0}
```

This means:
- London has an average price of 150.0.
- New York has an average price of 190.0, and so on.

#### Step 2: Encoding the Categorical Variable
Now, we map the cities to their corresponding mean prices using the `map()` function. This replaces the city names with the mean price values, creating a numerical relationship between the `city` and `price`.

```python
df['city_encoded'] = df['city'].map(mean_price)
df[['price', 'city_encoded']]
```

This generates a new column `city_encoded`, where:
- Cities like "New York" are replaced by their average price (190.0),
- "Paris" by 310.0, etc.

#### Result:
```plaintext
   price  city_encoded
0    200         190.0
1    150         150.0
2    300         310.0
3    250         250.0
4    180         190.0
5    320         310.0
```

So, now the cities have been encoded based on the mean price of that city. This kind of encoding captures the relationship between the category and the target variable in a way that can be more informative for predictive models.

### Why Use Target Guided Ordinal Encoding?
- **Large number of categories**: It is useful when a categorical variable has a lot of unique values (like city names, product IDs).
- **Monotonic relationship**: It creates a relationship between the category and the target variable (price, in this case). This is important because machine learning algorithms typically perform better when there’s a numerical and ordered relationship between features and target variables.

### Real-World Example:
Suppose you are working on a house price prediction problem, and you have a `neighborhood` variable. Instead of just encoding neighborhoods arbitrarily, you can replace them with the average house price of that neighborhood. This helps your model understand which neighborhoods tend to be more expensive and which are less so.



# EDA (Exploratory Data Analysis) on the Red Wine Dataset:

### 1. Data Overview
You are working with a dataset containing 1599 rows and 12 columns. The dataset provides information about the physicochemical properties of wine and its quality rating (output variable). Here's what you've done so far:

- **First look at the data**: By loading the data using `df.head()`, you previewed the first five rows, which include variables such as acidity, alcohol content, and the quality score of the wine.
  
### 2. Data Structure
You used the `df.info()` method to get a quick summary of the dataset's structure, confirming:
- **1599 total entries**.
- **11 columns with float values**, and **1 column with integer values** (`quality`).
- **No missing values** are present in the dataset.

### 3. Descriptive Statistics
You ran `df.describe()` to calculate statistics like:
- **Mean values** for the variables, e.g., the average alcohol content is around **10.42**.
- **Standard deviation** shows variability, e.g., the total sulfur dioxide has a high variance (`32.9`).
- **Min and Max values** help in identifying the range for each variable.

### 4. Shape of the Data
By running `df.shape`, you confirmed the original shape of the dataset as **(1599, 12)**, i.e., 1599 records and 12 columns.

### 5. Unique Values in Quality
You explored the unique values in the `quality` column using `df['quality'].unique()`. The quality scores are between **3** and **8**, representing the overall sensory rating of the wine.

### 6. Handling Missing Values
You checked for missing values using `df.isnull().sum()` and found that there are no missing values in the dataset.

### 7. Identifying Duplicates
You found 240 duplicate rows using `df[df.duplicated()]`. After dropping these duplicate records using `df.drop_duplicates()`, the new shape of the data is **(1359, 12)**.

### 8. Correlation Matrix
You used `df.corr()` to compute pairwise correlation coefficients between variables. For example:
- **Fixed acidity** and **density** show a strong positive correlation (0.67).
- **pH** and **fixed acidity** are negatively correlated (-0.68), indicating wines with higher fixed acidity tend to have lower pH values.
In the line:

```python
sns.heatmap(df.corr(), annot=True)
```

The parameter `annot=True` means that **the correlation values will be displayed** in each cell of the heatmap.

Here's what happens when you use `annot=True`:
- Normally, the heatmap just shows colors to represent the correlation between variables.
- When `annot=True`, it will **add the actual correlation numbers** (like 0.5, -0.8, etc.) on top of the colored cells so you can see the exact correlation value for each pair of variables.

### Example:
Let’s say the correlation matrix looks like this:

```
      A     B     C
A  1.00  0.80  0.60
B  0.80  1.00  0.20
C  0.60  0.20  1.00
```

Without `annot=True`, you'd only see a heatmap with colors, but **no numbers** in the cells.

With `annot=True`, you will also see the **correlation values** written inside each cell, making it easy to interpret the exact values:

```
   --------------------
   | 1.00 | 0.80 | 0.60|
   | 0.80 | 1.00 | 0.20|
   | 0.60 | 0.20 | 1.00|
   --------------------
```
---

### Next Steps for EDA
Here are additional steps you can include for a more comprehensive EDA:

1. **Visualization**: Visualize correlations using a heatmap, and plot distributions of individual variables.
   ```python
   import seaborn as sns
   import matplotlib.pyplot as plt

   plt.figure(figsize=(12, 8))
   sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
   plt.show()
   ```

2. **Data Distribution**: Plot histograms of numeric columns to understand the data distribution.
   ```python
   df.hist(bins=20, figsize=(14, 10))
   plt.show()
   ```

3. **Quality vs Features**: Use box plots to explore how each feature relates to the `quality` score.
   ```python
   sns.boxplot(x='quality', y='alcohol', data=df)
   plt.show()
   ```


### Example DataFrame
Imagine we have a small dataset with some duplicate rows. Here's how it might look:

```python
import pandas as pd

# Creating a small DataFrame
data = {'Name': ['John', 'Mike', 'Sara', 'Mike', 'John'],
        'Age': [25, 30, 22, 30, 25],
        'City': ['NY', 'LA', 'SF', 'LA', 'NY']}

df = pd.DataFrame(data)
print(df)
```

Output:
```
   Name  Age City
0  John   25   NY
1  Mike   30   LA
2  Sara   22   SF
3  Mike   30   LA
4  John   25   NY
```

Notice that row 0 and row 4 are the same (`John`, `25`, `NY`), and row 1 and row 3 are also the same (`Mike`, `30`, `LA`). These are duplicate records.

### Code Explanation

#### 1. `df[df.duplicated()]`
This line **finds** the duplicate rows in the DataFrame.

- `df.duplicated()` checks each row and returns `True` if the row is a duplicate (i.e., it has appeared earlier in the DataFrame).
- `df[df.duplicated()]` shows only the rows where `duplicated()` is `True`.

For our example, it would return:
```python
# Finding duplicates
df[df.duplicated()]
```

Output:
```
   Name  Age City
3  Mike   30   LA
4  John   25   NY
```
This tells us that row 3 and row 4 are duplicates of earlier rows.

#### 2. `df.drop_duplicates(inplace=True)`
This line **removes** the duplicate rows.

- `drop_duplicates()` removes all rows that are marked as duplicates (keeping only the first occurrence).
- `inplace=True` modifies the original DataFrame directly.

If we run it on our example:

```python
# Removing duplicates
df.drop_duplicates(inplace=True)

print(df)
```

Output:
```
   Name  Age City
0  John   25   NY
1  Mike   30   LA
2  Sara   22   SF
```

Now, the duplicates are removed, and only the unique rows remain.

---

### Summary
1. `df[df.duplicated()]`: Finds and displays the duplicate rows.
2. `df.drop_duplicates(inplace=True)`: Removes duplicate rows from the DataFrame.

Let's break down this part of the code:

### 1. **Bar Plot for Wine Quality**

```python
df.quality.value_counts().plot(kind='bar')
plt.xlabel("Wine Quality")
plt.ylabel("Count")
plt.show()
```

- **df.quality.value_counts()**: This counts the number of times each quality score appears in the dataset.
- **plot(kind='bar')**: This creates a bar plot of the wine quality counts.
- **plt.xlabel("Wine Quality")** and **plt.ylabel("Count")**: These label the x-axis (Wine Quality) and y-axis (Count), respectively.
- **plt.show()**: Displays the plot.

The code above is creating a bar plot to visualize the distribution of the **wine quality scores**. If the dataset is imbalanced, some quality scores will have significantly more counts than others.

### 2. **Histogram for Each Column**

```python
for column in df.columns:
    sns.histplot(df[column], kde=True)
```

- **for column in df.columns**: This loops through each column in the dataframe.
- **sns.histplot(df[column], kde=True)**: This creates a histogram for each column in the dataframe with an overlay of a Kernel Density Estimate (KDE) curve to show the distribution.

This is a quick way to visualize the **distribution of all the columns** in your dataset. The KDE curve provides a smoothed version of the histogram to show the general shape of the data.

### 3. **Histogram for the 'alcohol' Column**

```python
sns.histplot(df['alcohol'])
```

- **sns.histplot(df['alcohol'])**: This creates a histogram specifically for the **'alcohol' column** in the dataset to visualize the distribution of alcohol content in the wines.

### Summary
- The **first plot** is a bar chart that shows how many wines have each quality score.
- The **looped histograms** help you visualize the distribution of every column in the dataset, including features like pH, acidity, etc.
- The **third plot** focuses on the distribution of the alcohol content in the wine.
Here’s an extended version of your Jupyter notebook for **Exploratory Data Analysis (EDA) and Feature Engineering** for **Flight Price Prediction**. I'll help you explore the dataset and engineer relevant features step by step.
  


# FLIGHT PRICE DATASET
### **1. Importing Libraries**
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```

### **2. Loading Dataset**
```python
# Load the dataset
df = pd.read_excel('flight_price.xlsx')

# Display the first few rows of the dataset
df.head()
```

### **3. Data Overview**
Check the basic information about the dataset to understand its structure and contents.
```python
df.info()
df.describe()
```

### **4. Feature Engineering**

#### 4.1 Splitting `Date_of_Journey` into Day, Month, and Year
We will extract the date, month, and year from the `Date_of_Journey` column.
```python
# Splitting 'Date_of_Journey' into 'Date', 'Month', and 'Year'
df['Date'] = df['Date_of_Journey'].str.split('/').str[0].astype(int)
df['Month'] = df['Date_of_Journey'].str.split('/').str[1].astype(int)
df['Year'] = df['Date_of_Journey'].str.split('/').str[2].astype(int)

# Drop the 'Date_of_Journey' column as it's no longer needed
df.drop('Date_of_Journey', axis=1, inplace=True)

# Display the updated dataframe
df.head()
```
Sure! This line of code is part of **feature engineering**, which involves creating new features (columns) from existing ones to make your dataset more useful for analysis or modeling.

#### Explanation

1. **DataFrame**: `df` is a pandas DataFrame, which is a 2-dimensional labeled data structure like a table.
2. **Original Column**: The column `Date_of_Journey` contains dates in the format `DD/MM/YYYY`, meaning the day, month, and year are separated by slashes (`/`).
3. **Splitting the String**: The `.str.split('/')` method is used to split the date string into a list of components based on the `/` separator.
4. **Accessing List Elements**:
   - `.str[0]` accesses the first element (day).
   - `.str[1]` accesses the second element (month).
   - `.str[2]` accesses the third element (year).
5. **Creating New Columns**: New columns `Date`, `Month`, and `Year` are created in the DataFrame, storing the respective components from `Date_of_Journey`.

#### Example

Let's say your DataFrame `df` looks like this before the operation:

| Date_of_Journey |
|------------------|
| 25/12/2023       |
| 01/01/2024       |
| 15/05/2024       |

After running the code, the DataFrame `df` will be modified to:

| Date_of_Journey | Date | Month | Year |
|------------------|------|-------|------|
| 25/12/2023       | 25   | 12    | 2023 |
| 01/01/2024       | 01   | 01    | 2024 |
| 15/05/2024       | 15   | 05    | 2024 |

### 4.2 Processing `Arrival_Time` into `Arrival_Hour` and `Arrival_Minute`

#### 1. **Extracting Just the Time**

**Code:**
```python
df['Arrival_Time'] = df['Arrival_Time'].apply(lambda x: x.split(' ')[0])
```

- **What it does**: This line takes each value in the `Arrival_Time` column and keeps only the time part, removing anything after a space (like "AM" or "PM").
  
- **How it works**:
  - **`x.split(' ')`**: This splits the string into parts based on spaces.
    - For example: 
      - If `x` is `"10:30 AM"`, it becomes `["10:30", "AM"]`.
  - **`[0]`**: This selects the first part of the split, which is the time.
    - So, it gets `"10:30"` from the split result.

**Example:**
- Before: 
  - `df['Arrival_Time']` = `["10:30 AM", "12:45 PM", "08:15 AM"]`
- After: 
  - `df['Arrival_Time']` = `["10:30", "12:45", "08:15"]`

#### 2. **Extracting the Hour**

**Code:**
```python
df['Arrival_hour'] = df['Arrival_Time'].str.split(':').str[0]
```

- **What it does**: This line creates a new column called `Arrival_hour` that keeps only the hour part from the `Arrival_Time` column.
  
- **How it works**:
  - **`.str.split(':')`**: This splits the time string at the colon `:`.
    - For example: 
      - If the time is `"10:30"`, it becomes `["10", "30"]`.
  - **`.str[0]`**: This selects the first part of the split, which is the hour.
    - So, it gets `"10"` from the split result.

**Example:**
- Before: 
  - `df['Arrival_Time']` = `["10:30", "12:45", "08:15"]`
- After: 
  - `df['Arrival_hour']` = `["10", "12", "08"]`

#### 3. **Extracting the Minutes**

**Code:**
```python
df['Arrival_min'] = df['Arrival_Time'].str.split(':').str[1]
```

- **What it does**: This line creates another new column called `Arrival_min` that keeps only the minutes part from the `Arrival_Time` column.

- **How it works**:
  - **`.str.split(':')`**: Just like before, it splits the time string at the colon `:`.
    - For example: 
      - If the time is `"10:30"`, it becomes `["10", "30"]`.
  - **`.str[1]`**: This selects the second part of the split, which is the minutes.
    - So, it gets `"30"` from the split result.

**Example:**
- Before: 
  - `df['Arrival_Time']` = `["10:30", "12:45", "08:15"]`
- After: 
  - `df['Arrival_min']` = `["30", "45", "15"]`

#### Summary of All Steps

After applying all three lines of code, here’s how the DataFrame changes:

**Initial DataFrame**:
| Arrival_Time  |
|----------------|
| 10:30 AM      |
| 12:45 PM      |
| 08:15 AM      |

**After the First Line**:
| Arrival_Time  |
|----------------|
| 10:30         |
| 12:45         |
| 08:15         |

**After the Second Line (with Arrival_hour)**:
| Arrival_Time  | Arrival_hour |
|----------------|--------------|
| 10:30         | 10           |
| 12:45         | 12           |
| 08:15         | 08           |

**After the Third Line (with Arrival_min)**:
| Arrival_Time  | Arrival_hour | Arrival_min |
|----------------|--------------|--------------|
| 10:30         | 10           | 30           |
| 12:45         | 12           | 45           |
| 08:15         | 08           | 15           |

#### There are several other ways to achieve the same result without using the `apply` method with a `lambda` function. Here are a few alternatives:

### 1. **Using `.str` Methods Directly**

You can use the string accessor `.str` directly to manipulate the strings in the DataFrame column. 

**Code:**
```python
df['Arrival_Time'] = df['Arrival_Time'].str.split(' ').str[0]
```

- **Explanation**: This line does exactly the same thing as your original line. It splits the strings in the `Arrival_Time` column by spaces and selects the first part (the time).

### 2. **Using Regular Expressions with `.str.replace()`**

You can use regular expressions to remove the part after the space.

**Code:**
```python
df['Arrival_Time'] = df['Arrival_Time'].str.replace(r'\s.*', '', regex=True)
```

- **Explanation**: This line uses a regular expression `r'\s.*'` to match a space followed by any characters. It replaces that match with an empty string, effectively removing everything after the space.

### 3. **Using `.str.extract()` with Regular Expressions**

You can also use the `str.extract()` method with a regex pattern to capture just the time part.

**Code:**
```python
df['Arrival_Time'] = df['Arrival_Time'].str.extract(r'(\d{1,2}:\d{2})')
```

- **Explanation**: This uses the regex pattern `(\d{1,2}:\d{2})` to match and capture the time format (like `10:30`). The parentheses indicate that we want to keep only the part that matches this pattern.

### 4. **Using `.str.slice()`**

If you know the position of the time, you can use `str.slice()` to get the first part.

**Code:**
```python
df['Arrival_Time'] = df['Arrival_Time'].str.slice(0, 5)
```

- **Explanation**: This slices the string from the beginning (index `0`) to the 5th character, which would cover the time format `HH:MM` in most cases.

### Example Comparison

Assuming you have a DataFrame like this:

| Arrival_Time  |
|----------------|
| 10:30 AM      |
| 12:45 PM      |
| 08:15 AM      |

Using any of the methods above would yield:

| Arrival_Time  |
|----------------|
| 10:30         |
| 12:45         |
| 08:15         |

#### 4.3 Processing `Dep_Time` into `Departure_Hour` and `Departure_Minute`
Similarly, process the `Dep_Time` into hour and minute for better modeling.
```python
# Splitting 'Dep_Time' into 'Departure_Hour' and 'Departure_Minute'
df['Departure_Hour'] = df['Dep_Time'].str.split(':').str[0].astype(int)
df['Departure_Minute'] = df['Dep_Time'].str.split(':').str[1].astype(int)

# Dropping 'Dep_Time' column as it's no longer needed
df.drop('Dep_Time', axis=1, inplace=True)

# Display the updated dataframe
df.head()
```

#### 4.4 Processing `Duration` into `Duration_Hours` and `Duration_Minutes`
Extract hours and minutes from the `Duration` column to improve feature engineering.
```python
# Handling the 'Duration' column
duration = df['Duration'].str.split(' ')

# Splitting the 'Duration' into hours and minutes
df['Duration_Hours'] = duration.apply(lambda x: int(x[0].replace('h', '')) if 'h' in x[0] else 0)
df['Duration_Minutes'] = duration.apply(lambda x: int(x[1].replace('m', '')) if len(x) > 1 else 0)

# Dropping the 'Duration' column as it's no longer needed
df.drop('Duration', axis=1, inplace=True)

# Display the updated dataframe
df.head()
```

### 1. **Finding Unique Values**
```python
df['Total_Stops'].unique()
```
- **What it does**: This line retrieves all the unique values in the `Total_Stops` column of the DataFrame.
- **Why we do it**: It helps us understand what options we have in this column (like "non-stop," "1 stop," etc.) and see if there are any missing values (`nan`).

### Example Output:
```plaintext
array(['non-stop', '2 stops', '1 stop', '3 stops', nan, '4 stops'],
      dtype=object)
```
- This tells us the different types of stop counts in the dataset.

---

### 2. **Identifying Missing Values**
```python
df[df['Total_Stops'].isnull()]
```
- **What it does**: This line filters the DataFrame to show only the rows where the `Total_Stops` value is `null` (missing).
- **Why we do it**: To find out how many records are missing data for `Total_Stops`, which helps in cleaning the dataset.

### Example Output:
```plaintext
    Airline    Source    Destination   Route     Duration   Total_Stops  Additional_Info  Price  Date  Month  Year  Arrival_hour  Arrival_min  Departure_hour  Departure_min
9039 Air India   Delhi     Cochin       NaN     23h 40m       NaN         No info         7480    6    5    2019       9             25            9             45
```
- Here, you can see the specific entry with missing stops.

---

### 3. **Finding the Most Common Value**
```python
df['Total_Stops'].mode()
```
- **What it does**: This line finds the most common value in the `Total_Stops` column.
- **Why we do it**: To know which stop count is the most frequent; this can help in deciding how to fill in the missing values.

### Example Output:
```plaintext
0    1 stop
Name: Total_Stops, dtype: object
```
- In this case, "1 stop" is the most common.

---

### 4. **Mapping Values**
```python
df['Total_Stops'] = df['Total_Stops'].map({'non-stop': 0, '1 stop': 1, '2 stops': 2, '3 stops': 3, '4 stops': 4, np.nan: 1})
```
- **What it does**: This line replaces the text values in `Total_Stops` with numeric values (e.g., "non-stop" becomes 0, "1 stop" becomes 1, etc.). It also replaces any `nan` values with 1.
- **Why we do it**: Numeric values are easier to work with when analyzing data or building models. It allows us to quantify the number of stops.

### Example Mapping:
- "non-stop" → 0
- "1 stop" → 1
- "2 stops" → 2
- "3 stops" → 3
- "4 stops" → 4
- `nan` → 1 (assuming we fill missing data with the most common value)

---

### 5. **Check for Remaining Missing Values**
```python
df[df['Total_Stops'].isnull()]
```
- **What it does**: This line again checks if there are still any `nan` values in the `Total_Stops` column after mapping.
- **Why we do it**: To ensure that all missing values have been addressed and that our data is clean.

### Summary:
- **Why are we doing all this?** 
  - To clean and prepare the data for analysis.
  - Understanding unique values helps identify potential issues (like missing data).
  - Replacing text with numbers makes it easier for analysis and machine learning models.


Let’s break this down step by step in a simple way!

### What's Happening Here?

The code you provided is using **OneHotEncoding** to convert categorical columns (`'Airline'`, `'Source'`, and `'Destination'`) into a numerical format. This is necessary because machine learning models can't handle text or categories directly—they need numbers. **OneHotEncoding** is a popular way to convert categorical data into numbers without losing the information about the categories.

Let’s explain each part:

---

### Step 1: Importing the OneHotEncoder
```python
from sklearn.preprocessing import OneHotEncoder
```
Here, you’re importing `OneHotEncoder` from the `sklearn.preprocessing` library. `OneHotEncoder` will be used to convert the text-based categorical columns into a numeric format.

---

### Step 2: Initializing the Encoder
```python
encoder = OneHotEncoder()
```
You are creating an instance of `OneHotEncoder()` and storing it in the variable `encoder`. This prepares the encoder for use.

---

### Step 3: Applying the Encoder
```python
encoder.fit_transform(df[['Airline','Source','Destination']]).toarray()
```
Here, you are applying the `encoder` to the DataFrame (`df`) columns `['Airline', 'Source', 'Destination']`. Let’s break this down:

- **`fit_transform()`**: This does two things: 
  1. **`fit`**: It learns the categories present in the columns (like which airlines, sources, and destinations exist).
  2. **`transform`**: It converts the categorical data into a numerical format.
  
- **`df[['Airline', 'Source', 'Destination']]`**: This selects the three columns (`'Airline'`, `'Source'`, `'Destination'`) that need to be encoded.

- **`.toarray()`**: This converts the transformed data into a numpy array. The result is a matrix of 0s and 1s.

#### Example:
Imagine the `'Airline'` column has three categories: `'IndiGo'`, `'Air India'`, and `'SpiceJet'`. After OneHotEncoding, it might look like this:

| Airline        | OneHotEncoded Columns (IndiGo, Air India, SpiceJet) |
|----------------|-----------------------------------------------------|
| IndiGo         | [1, 0, 0]                                           |
| Air India      | [0, 1, 0]                                           |
| SpiceJet       | [0, 0, 1]                                           |

Each category is converted into a separate column where:
- 1 means "yes, this is the category."
- 0 means "no, this isn’t the category."

The same idea applies to the `Source` and `Destination` columns.

---

### Step 4: Creating a DataFrame from the Encoded Array
```python
pd.DataFrame(encoder.fit_transform(df[['Airline','Source','Destination']]).toarray(), columns=encoder.get_feature_names_out())
```
Now, we are converting the resulting array from OneHotEncoding back into a **DataFrame** (a table) with proper column names. Here’s what each part does:

- **`pd.DataFrame()`**: This creates a new DataFrame (table) from the array.
  
- **`encoder.get_feature_names_out()`**: This method gives you the names of the newly created columns after encoding (e.g., `Airline_IndiGo`, `Airline_Air India`, `Source_Delhi`, etc.).

#### Result:
After this, you’ll get a new DataFrame where the text categories have been replaced by numbers (0s and 1s) in new columns.

---

### **Why Do We Use OneHotEncoder?**

- **Machine Learning Needs Numbers**: Algorithms can’t work with text, so we convert categories into numbers.
  
- **Avoiding Bias**: OneHotEncoding ensures that each category is treated equally. It avoids giving any particular category a "higher value" (as would happen with just labeling them with 1, 2, 3, etc.).

---

### Simple Example to Understand:
Imagine you have a small dataset like this:

| Airline  | Source  | Destination |
|----------|---------|-------------|
| IndiGo   | Delhi   | Mumbai      |
| Air India| Kolkata | Chennai     |

#### OneHotEncoding will convert it into:

| Airline_IndiGo | Airline_Air India | Source_Delhi | Source_Kolkata | Destination_Mumbai | Destination_Chennai |
|----------------|-------------------|--------------|----------------|--------------------|---------------------|
| 1              | 0                 | 1            | 0              | 1                  | 0                   |
| 0              | 1                 | 0            | 1              | 0                  | 1                   |

Each category is split into its own column with binary values.

---

### Summary:
- **OneHotEncoder** is converting text categories (`Airline`, `Source`, `Destination`) into numeric 0s and 1s.
- It creates new columns for each unique value (like `Airline_IndiGo`, `Airline_Air India`, etc.).
- This is needed because machine learning algorithms work with numbers, not text.

# EDA And Feature Engineering Of Google Play Store Dataset


#### 1. **Problem Statement and Objective**:
   - **Problem**: There are millions of apps available for download. The goal of this analysis is to identify the most popular category, the app with the largest number of installs, the app with the largest size, etc.
   - **Dataset**: It contains 20 columns and 10841 rows.

#### 2. **Steps in the Analysis**:
   - **Data Cleaning**
   - **Exploratory Data Analysis (EDA)**
   - **Feature Engineering**

#### 3. **Libraries Imported**:
   - **pandas** and **numpy**: Used for data manipulation and analysis.
   - **matplotlib** and **seaborn**: Visualization libraries.
   - **warnings**: To suppress warning messages.
   - `%matplotlib inline` makes sure plots appear in the notebook.

#### 4. **Loading the Data**:
   ```python
   df=pd.read_csv('https://raw.githubusercontent.com/krishnaik06/playstore-Dataset/main/googleplaystore.csv')
   ```
   - This loads the dataset directly from a URL.

#### 5. **First Look at the Data**:
   ```python
   df.head()
   ```
   - Displays the first 5 rows of the dataset. This helps get a feel for the data, its structure, and types of columns.

#### 6. **Shape of the Dataset**:
   ```python
   df.shape
   ```
   - Output: `(10841, 13)`
   - Indicates there are 10841 rows and 13 columns after loading.

#### 7. **Dataset Information**:
   ```python
   df.info()
   ```
   - **Output**: Shows the column names, number of non-null entries, and data types for each column.
   - Important Observations:
     - Some columns like `Rating`, `Type`, and `Android Ver` have missing values.
     - The dataset contains both **numerical** and **categorical** data.

#### 8. **Dataset Summary**:
   ```python
   df.describe()
   ```
   - Provides statistical summaries like `mean`, `min`, `max`, etc., for numerical columns.
   - For example, the average rating (`mean`) is 4.19, and the maximum value is 19 (likely an anomaly).

#### 9. **Missing Values Check**:
   ```python
   df.isnull().sum()
   ```
   - Shows the count of missing values in each column.
   - Key Insights:
     - `Rating` has 1474 missing values.
     - `Type` and `Content Rating` each have 1 missing value.
     - `Current Ver` has 8 missing values, while `Android Ver` has 3 missing values.

#### 10. **Insights and Observations**:
   - **Missing Values**: There are some missing values, especially in `Rating`. These missing values will likely need to be handled during data cleaning.

---

### 1. Checking Numeric Values in the `Reviews` Column
The first line checks if the values in the `Reviews` column are numeric.

```python
df['Reviews'].str.isnumeric().sum()
```
- **What it does:** It checks how many entries in the `Reviews` column are numeric (i.e., contain only numbers).
- **Why:** `Reviews` should be numeric, but sometimes there are non-numeric values (like `19.0`). We need to identify them to handle them later.

### 2. Filtering Non-Numeric Reviews
This line filters out rows where the `Reviews` column is not numeric.

```python
df[~df['Reviews'].str.isnumeric()]
```
- **What it does:** The `~` symbol negates the condition, meaning it selects rows where `Reviews` are not numeric.
- **Why:** It helps us find any entries that might cause problems when converting `Reviews` to integers. For instance, the entry with `19.0` in the `Reviews` column needs to be fixed.

### 3. Dropping a Specific Row with Non-Numeric Reviews
This line creates a copy of the dataframe and removes a problematic row (index `10472`).

```python
df_copy = df.copy()
df_copy = df_copy.drop(df_copy.index[10472])
```
- **What it does:** It drops the row at index `10472` because it contains invalid data (`19.0` in `Reviews`).
- **Why:** We can't convert the non-numeric value `19.0` to an integer, so we remove that row.

### 4. Converting the `Reviews` Column to Integer
Once the problematic row is removed, we can safely convert the `Reviews` column to an integer.

```python
df_copy['Reviews'] = df_copy['Reviews'].astype(int)
```
- **What it does:** Converts the `Reviews` column from a string (or object type) to an integer.
- **Why:** `Reviews` is supposed to represent the number of reviews, which should be a numeric type for analysis.

### 5. Checking Information of the DataFrame
This line gives us details about the dataframe, like the number of entries, non-null values, and data types.

```python
df_copy.info()
```
- **What it does:** Displays information about each column, such as the data type and whether there are any missing values.
- **Why:** It's a good practice to check the structure of the dataframe to ensure that the columns are of the correct data type.

### 6. Unique Values in the `Size` Column
The following line lists all unique values in the `Size` column.

```python
df_copy['Size'].unique()
```
- **What it does:** Displays the unique values in the `Size` column.
- **Why:** To identify if there are any non-standard or missing values in this column, like `Varies with device`.

### 7. Replacing Text in the `Size` Column
Here, we convert values from megabytes (M) or kilobytes (k) into a consistent numerical format.

```python
df_copy['Size'] = df_copy['Size'].str.replace('M', '000')
df_copy['Size'] = df_copy['Size'].str.replace('k', '')
```
- **What it does:** Replaces `M` with `000` to convert megabytes to kilobytes. For example, `19M` becomes `19000`. Similarly, it removes the `k` in kilobytes.
- **Why:** To standardize the size data for easier analysis later on.

### 8. Handling Missing Values in the `Size` Column
We replace values like `Varies with device` with `NaN` to mark them as missing values.

```python
df_copy['Size'] = df_copy['Size'].replace('Varies with device', np.nan)
```
- **What it does:** Replaces the value `Varies with device` with `NaN` (Not a Number).
- **Why:** `Varies with device` is not a numeric value, and we treat it as missing data.

### 9. Converting the `Size` Column to Float
After cleaning, we convert the `Size` column to a floating-point number type.

```python
df_copy['Size'] = df_copy['Size'].astype(float)
```
- **What it does:** Converts the `Size` column to `float` type for accurate analysis.
- **Why:** Some values in the `Size` column might be fractional, and floating-point numbers are more flexible than integers.

### 10. Checking Unique Values in the `Installs` Column
This command shows the unique values in the `Installs` column, which represent how many times the app was installed.

```python
df_copy['Installs'].unique()
```
- **What it does:** Lists all the unique values in the `Installs` column.
- **Why:** To identify the range of installation numbers and handle any irregular values.
- **What we did:** We cleaned up the `Reviews` and `Size` columns by converting them into appropriate numerical types and handled missing or invalid values.
- **Why we did it:** This process is important for data analysis because working with consistent, numeric data allows us to perform meaningful calculations and visualizations.


### Code Block:
```python
df_copy['Size'] = df_copy['Size'].replace('Varies with device', np.nan)
df_copy['Size'] = df_copy['Size'].astype(float)
```

### Step-by-Step Breakdown:

1. **Replace 'Varies with device' with `NaN`:**
   ```python
   df_copy['Size'] = df_copy['Size'].replace('Varies with device', np.nan)
   ```
   - **What it does:** 
     - The value `'Varies with device'` in the `Size` column is a string, and it means that the app's size changes based on the device it's installed on. Since this isn't a fixed numeric value, we replace it with `NaN` (Not a Number), which is Pandas' way of handling missing or undefined values.
   - **Why it's useful:** 
     - By replacing non-numeric strings like `'Varies with device'` with `NaN`, you mark them as missing data. This makes it easier to handle later during analysis or statistical operations.
     - For example, you can skip or fill `NaN` values when performing calculations like taking the mean or plotting data.

2. **Convert the `Size` Column to Float:**
   ```python
   df_copy['Size'] = df_copy['Size'].astype(float)
   ```
   - **What it does:** 
     - This line converts the entire `Size` column to `float` (floating-point numbers).
     - `NaN` is actually treated as a special floating-point value in Pandas, so you **can** convert a column containing `NaN` to `float`.
   - **Why it's useful:**
     - You are standardizing the `Size` column by converting it to a numeric type. This is important for numerical operations (e.g., finding the average size of apps).
     - Pandas treats `NaN` as a float-compatible value, so when you convert the column, `NaN` remains `NaN` but now fits into a column of floating-point numbers. Other valid numeric values (like app sizes in kilobytes or megabytes) are converted into `float`.

### Why Convert to `float` When There's `NaN`?
- **`NaN` is float-compatible:** In Pandas and NumPy, `NaN` is a valid floating-point value. When you perform operations on a `float` column that contains `NaN`, it simply ignores those missing values.
- **Why convert to `float` instead of `int`?:** If the `Size` column had any `NaN` values, converting it to an integer type would cause an error because `NaN` cannot be represented as an integer. However, `NaN` fits well with floating-point numbers, so converting the column to `float` ensures everything is handled correctly.

### Example:
```python
# Initial column with 'Varies with device'
Size
0        19M
1     2.5M
2    Varies with device
3        40M

# After replacing 'Varies with device' with NaN:
Size
0    19000.0
1     2500.0
2        NaN
3    40000.0
```

Here, `NaN` acts as a placeholder for missing or undefined values while allowing other numeric values to be stored as `float`.

### Summary:
- You can convert a column containing `NaN` to `float` because `NaN` is a floating-point value in Pandas.
- Replacing strings like `'Varies with device'` with `NaN` ensures you can work with numeric data, while converting the column to `float` allows for smooth numeric operations and avoids errors.


### Step 1: Remove Unwanted Characters

```python
chars_to_remove = ['+', ',', '$']
cols_to_clean = ['Installs', 'Price']

for item in chars_to_remove:
    for cols in cols_to_clean:
        df_copy[cols] = df_copy[cols].str.replace(item, '')
```

**What this does:**
- **Purpose:** The goal is to clean the `Installs` and `Price` columns by removing specific unwanted characters (`+`, `,`, `$`) that may interfere with data type conversion.
- **How it works:** 
  - It loops through each character in `chars_to_remove` and each column in `cols_to_clean`.
  - The `str.replace()` method is used to replace occurrences of each unwanted character with an empty string, effectively removing it.

### Step 2: Check Unique Values

```python
df_copy['Price'].unique()
```

**What this does:**
- **Purpose:** This line checks the unique values in the `Price` column after the cleaning process. This helps ensure that unwanted characters have been removed.
- **Result:** You’ll see an array of unique price values (as strings) without the special characters.

### Step 3: Convert Columns to Appropriate Data Types

```python
df_copy['Installs'] = df_copy['Installs'].astype('int')
df_copy['Price'] = df_copy['Price'].astype('float')
```

**What this does:**
- **Purpose:** After cleaning the data, it's essential to convert the columns to the correct data types for further analysis.
- **How it works:**
  - The `Installs` column is converted to integers, which allows for numerical calculations.
  - The `Price` column is converted to floats, as prices can have decimal values.

### Step 4: Handle Last Updated Feature

```python
df_copy['Last Updated'] = pd.to_datetime(df_copy['Last Updated'])
df_copy['Day'] = df_copy['Last Updated'].dt.day
df_copy['Month'] = df_copy['Last Updated'].dt.month
df_copy['Year'] = df_copy['Last Updated'].dt.year
```

**What this does:**
- **Purpose:** Convert the `Last Updated` column to a datetime format to extract specific date components (day, month, year).
- **How it works:**
  - `pd.to_datetime()` converts the `Last Updated` string to a datetime object.
  - The `dt` accessor allows you to extract the day, month, and year from the datetime object, creating new columns for each.

### Step 5: Check DataFrame Info

```python
df_copy.info()
```

**What this does:**
- **Purpose:** This command displays the summary of the DataFrame, including the number of entries, columns, data types, and memory usage.
- **Result:** You can verify that the data types are correct and that new columns (Day, Month, Year) have been successfully added.

### Step 6: View the Cleaned Data

```python
df_copy.head()
```

**What this does:**
- **Purpose:** This command displays the first few rows of the cleaned DataFrame for a quick review.
- **Result:** You can visually inspect the changes made to the DataFrame and ensure that the data looks as expected.

### Step 7: Save the Cleaned Data

```python
df_copy.to_csv('data/google_cleaned.csv')
```

**What this does:**
- **Purpose:** This line saves the cleaned DataFrame to a CSV file for future use or analysis.
- **Result:** You now have a persistent copy of your cleaned data that can be easily accessed later.

### Summary

- **Cleaning Data:** This process involves removing unwanted characters and converting data types, which is crucial for accurate analysis.
- **Extracting Date Components:** Handling date data appropriately allows for more insightful analysis regarding trends over time.
- **Final Output:** Saving the cleaned DataFrame ensures you have a ready-to-use dataset for your next steps in analysis or modeling.
The line of code `df_copy[df_copy.duplicated('App')].shape` is used to find and count duplicate entries in the `App` column of the DataFrame `df_copy`. 

### Breakdown:

- **`df_copy.duplicated('App')`**: Checks for duplicate rows based on the `App` column and returns a boolean Series.
- **`df_copy[...]`**: Filters the DataFrame to include only the duplicated rows.
- **`.shape`**: Returns a tuple with the dimensions of the resulting DataFrame (number of rows and columns).

### Meaning:

- The output, like `(10, 13)`, indicates **10 duplicate rows** based on the `App` column, with a total of **13 columns** in the DataFrame. 


### Breakdown of the Code:

1. **Dropping Duplicates**:
   ```python
   df_copy = df_copy.drop_duplicates(subset=['App'], keep='first')
   ```
   - This line removes duplicate entries in the `App` column, keeping only the first occurrence. The DataFrame `df_copy` now has a shape of `(9659, 16)`, meaning there are **9,659 rows** and **16 columns** after dropping duplicates.

2. **Identifying Numeric and Categorical Features**:
   ```python
   numeric_features = [feature for feature in df_copy.columns if df_copy[feature].dtype != 'O']
   categorical_features = [feature for feature in df_copy.columns if df_copy[feature].dtype == 'O']
   ```
   - **`numeric_features`**: A list comprehension that creates a list of column names where the data type is not an object (`'O'` is the code for object types in pandas, which usually means strings).
   - **`categorical_features`**: A list comprehension that creates a list of column names where the data type is an object.

3. **Printing the Results**:
   ```python
   print('We have {} numerical features : {}'.format(len(numeric_features), numeric_features))
   print('\nWe have {} categorical features : {}'.format(len(categorical_features), categorical_features))
   ```
   - These lines print out the counts and names of the identified numerical and categorical features.

### Output Explanation:

- **Numerical Features**:
  - There are **9 numerical features**: `['Rating', 'Reviews', 'Size', 'Installs', 'Price', 'Last Updated', 'Day', 'Month', 'Year']`.
- **Categorical Features**:
  - There are **7 categorical features**: `['App', 'Category', 'Type', 'Content Rating', 'Genres', 'Current Ver', 'Android Ver']`.


### 3.2 Feature Information

This section provides a brief description of what each feature (column) in your dataset represents:

- **App**: The name of the application.
- **Category**: The type or category the app belongs to (like games, education, etc.).
- **Rating**: How well the app is rated on the Play Store.
- **Reviews**: The total number of reviews the app has received.
- **Size**: The storage space the app occupies on a device.
- **Installs**: How many times the app has been downloaded.
- **Type**: Indicates whether the app is free or paid.
- **Price**: The cost of the app (0 if it is free).
- **Content Rating**: The target audience for the app (like everyone, teens, etc.).
- **Genres**: The specific genre the app falls under (like action, adventure, etc.).
- **Last Updated**: The last date the app was updated.
- **Current Ver**: The latest version of the application available.
- **Android Ver**: The minimum version of Android required to use the app.

### Proportion of Count Data on Categorical Columns

This section looks at the distribution of the categorical features in the dataset:

1. **Loop through Categorical Features**:
   ```python
   for col in categorical_features:
       print(df[col].value_counts(normalize=True)*100)
       print('---------------------------')
   ```
   - This code goes through each categorical column and calculates the percentage of each unique value in that column.

2. **Output Example**:
   - For the `App` feature:
     - It shows how many times each app appears in the dataset, expressed as a percentage. For example, "ROBLOX" appears 0.08% of the time.
   - For the `Category` feature:
     - It shows the percentage of apps in each category, such as "FAMILY" (18.19%) and "GAME" (10.55%).
   - The same applies to other features like `Type`, `Content Rating`, `Genres`, `Current Ver`, and `Android Ver`.

### Proportion of Count Data on Numerical Columns

Next, the code visualizes the distribution of the numerical features in the dataset:

1. **Setting Up the Plot**:
   ```python
   plt.figure(figsize=(15, 15))
   plt.suptitle('Univariate Analysis of Numerical Features', fontsize=20, fontweight='bold', alpha=0.8, y=1.)
   ```
   - This creates a large figure to display multiple plots and sets a title for it.

2. **Plotting Each Numerical Feature**:
   ```python
   for i in range(0, len(numeric_features)):
       plt.subplot(5, 3, i+1)
       sns.kdeplot(x=df_copy[numeric_features[i]], shade=True, color='r')
       plt.xlabel(numeric_features[i])
       plt.tight_layout()
   ```
   - For each numerical feature, it creates a separate subplot (small plot).
   - It uses `sns.kdeplot` to draw a Kernel Density Estimate (KDE) plot, which shows the distribution of data points. This helps visualize where values are concentrated.
   - The `shade=True` argument fills the area under the curve, making it easier to see.

### Summary

- The code first explains what each feature in the dataset represents.
- It then calculates and prints the distribution of categorical features, showing what percentage of the total each category represents.
- Finally, it visualizes the distribution of numerical features, helping to understand how the values are spread across those features.


### Importance:

- **Data Exploration**: Understanding the types of features helps in choosing appropriate analysis techniques, visualizations, and machine learning models.
- **Numerical Features**: Suitable for statistical analysis and numerical modeling.
- **Categorical Features**: Useful for grouping, counting, or as inputs in classification models.
Univariate analysis of numerical features focuses on understanding individual numerical variables in a dataset. Here’s why it’s useful:

1. **Distribution**: It shows how values are spread (e.g., normal, skewed), helping you understand the overall pattern.

2. **Outliers**: It helps identify unusual values that differ significantly from the rest, which can affect analysis.

3. **Central Tendency**: You can determine average values (mean, median) and the range (min and max), giving insights into the feature's behavior.

4. **Guidance for Modeling**: Insights from the analysis can inform decisions on data preprocessing and model selection, leading to better performance.

5. **Hypothesis Testing**: It supports formulating and testing hypotheses about relationships in the data.
Here's a simplified breakdown of your observations and analysis:

### Observations
1. **Skewness in Numerical Features**:
   - **Left Skewed**: **Rating** and **Year** distributions have a longer tail on the left, indicating most apps have higher ratings and are from more recent years.
   - **Right Skewed**: **Reviews**, **Size**, **Installs**, and **Price** distributions have a longer tail on the right, showing that a few apps have very high reviews, size, installs, or prices compared to the majority.

2. **Categorical Features Analysis**:
   - A count plot is created for categorical features like **Type** and **Content Rating** to visualize how many apps fall into each category.
   - This helps identify trends in app types (e.g., free vs. paid) and content ratings (e.g., suitable for everyone or age-specific).

### Most Popular App Category
- By using the `value_counts()` method on the **Category** column, you plot a pie chart to visualize the distribution of app categories in the Play Store.
- **Findings**: 
   - The most popular categories are **Family**, **Games**, and **Tools**. 
   - There are fewer apps in categories like **Beauty**, **Comics**, **Arts**, and **Weather**, indicating these categories have less competition in the Play Store.

### Visualizations
- **Pie Chart**: The pie chart illustrates the proportion of apps across different categories, making it easy to see which categories are most or least common.


### Code Breakdown
1. **Counting Categories**:
   ```python
   category = pd.DataFrame(df_copy['Category'].value_counts())
   category.rename(columns={'Category': 'Count'}, inplace=True)
   ```
   - This code creates a DataFrame called `category` that counts how many apps belong to each category in the `df_copy` DataFrame.
   - The column is renamed from "Category" to "Count" for clarity.

2. **Bar Plot Visualization**:
   ```python
   plt.figure(figsize=(15, 6))
   sns.barplot(x=category.index[:10], y='Count', data=category[:10], palette='hls')
   plt.title('Top 10 App Categories')
   plt.xticks(rotation=90)
   plt.show()
   ```
   - A bar plot is generated to visualize the top 10 app categories based on their counts.
   - The x-axis shows the app categories, and the y-axis represents the number of apps in each category. The bars are colored using the 'hls' palette for better aesthetics.

### Insights
- **Top Categories**:
  - **Family**: The Family category has the highest number of apps, making up **18%** of all apps.
  - **Games**: The Games category follows with **11%** of apps.

- **Least Popular Category**:
  - **Beauty**: This category has the fewest apps, contributing to less than **1%** of the total apps.
Let’s break down the code step by step and explain it in a simple way, along with an example for better understanding.

### Code Breakdown

1. **Grouping and Summing Installations by Category**:
   ```python
   df_cat_installs = df_copy.groupby(['Category'])['Installs'].sum().sort_values(ascending=False).reset_index()
   ```
   - **`df_copy.groupby(['Category'])`**: This part groups the DataFrame `df_copy` by the 'Category' column. This means that all apps in the same category are considered together.
   - **`['Installs'].sum()`**: For each category, it sums up the total number of installations from all apps in that category. 
   - **`.sort_values(ascending=False)`**: This sorts the resulting DataFrame in descending order, so the category with the highest total installations comes first.
   - **`.reset_index()`**: This resets the index of the DataFrame after sorting, making it easier to work with.

   **Example**: 
   If `df_copy` has the following installation numbers:

   | Category | Installs |
   |----------|----------|
   | Game     | 10,000   |
   | Game     | 20,000   |
   | Family   | 5,000    |
   | Family   | 15,000   |

   The resulting DataFrame (`df_cat_installs`) will look like this:

   | Category | Installs |
   |----------|----------|
   | Game     | 30,000   |
   | Family   | 20,000   |

2. **Converting Installations to Billions**:
   ```python
   df_cat_installs.Installs = df_cat_installs.Installs / 1000000000  # converting into billions
   ```
   - This line converts the number of installations from absolute numbers to billions. It divides the total installations by 1,000,000,000 to make the numbers more manageable and easier to read.

   **Example**: 
   If the total installations for 'Game' were 30,000,000, after this line, it would be represented as 0.03 billion.

3. **Selecting the Top 10 Categories**:
   ```python
   df2 = df_cat_installs.head(10)
   ```
   - This takes the top 10 categories with the highest number of installations and stores them in a new DataFrame called `df2`.

4. **Creating a Bar Plot**:
   ```python
   plt.figure(figsize=(14, 10))
   sns.set_context("talk")
   sns.set_style("darkgrid")

   ax = sns.barplot(x='Installs', y='Category', data=df2)
   ax.set_xlabel('No. of Installations in Billions')
   ax.set_ylabel('')
   ax.set_title("Most Popular Categories in Play Store", size=20)
   plt.text(0.5, 1.0, 'Most Popular Categories in Play Store')
   plt.show()
   ```
   - **`plt.figure(figsize=(14, 10))`**: This sets the size of the plot to be 14 units wide and 10 units tall.
   - **`sns.set_context("talk")`** and **`sns.set_style("darkgrid")`**: These lines set the visual style of the plot to make it look better and easier to read.
   - **`sns.barplot(...)`**: This creates a bar plot with 'Installs' on the x-axis and 'Category' on the y-axis using the data from `df2`.
   - **`ax.set_xlabel(...)`** and **`ax.set_ylabel('')`**: These lines label the x-axis and clear the y-axis label.
   - **`ax.set_title(...)`**: This sets the title of the plot.
   - **`plt.text(...)`**: This adds text to the plot, providing additional context about the data being shown.
   - **`plt.show()`**: This displays the plot.

### Insights
- **Insights**:
  - The code concludes that the "GAME" category has the most installations, totaling almost **35 billion installations**, making it the most popular category in the Google Play Store.
Let’s break down the code step by step to understand how it identifies the top 5 most installed apps in each of the popular categories specified. 

### Code Breakdown

1. **Grouping and Summing Installations by Category and App**:
   ```python
   dfa = df_copy.groupby(['Category', 'App'])['Installs'].sum().reset_index()
   ```
   - **`df_copy.groupby(['Category', 'App'])`**: This groups the DataFrame `df_copy` by both 'Category' and 'App'. This means that for each app within a category, the data will be combined.
   - **`['Installs'].sum()`**: It calculates the total number of installations for each app in each category.
   - **`.reset_index()`**: This resets the index of the DataFrame to make it easier to read.

   **Example**:
   If `df_copy` contains the following data:

   | Category     | App               | Installs |
   |--------------|-------------------|----------|
   | Game         | Subway Surfers     | 1000000  |
   | Game         | Candy Crush Saga   | 1500000  |
   | Communication | Hangouts          | 500000   |
   | Communication | WhatsApp          | 2000000  |

   The resulting DataFrame (`dfa`) will look like this after summing installations:

   | Category       | App               | Installs |
   |----------------|-------------------|----------|
   | Game           | Subway Surfers     | 1000000  |
   | Game           | Candy Crush Saga   | 1500000  |
   | Communication  | Hangouts          | 500000   |
   | Communication  | WhatsApp          | 2000000  |

2. **Sorting Installations**:
   ```python
   dfa = dfa.sort_values('Installs', ascending=False)
   ```
   - This sorts the `dfa` DataFrame by the 'Installs' column in descending order, so the apps with the most installations appear first.

3. **Defining Popular Categories**:
   ```python
   apps = ['GAME', 'COMMUNICATION', 'PRODUCTIVITY', 'SOCIAL']
   ```
   - Here, we define a list called `apps` that includes the categories we are interested in analyzing.

4. **Setting Plot Context and Style**:
   ```python
   sns.set_context("poster")
   sns.set_style("darkgrid")
   plt.figure(figsize=(40, 30))
   ```
   - These lines set the visual context and style for the plots. The figure size is set to be very large for better visibility of details.

Let’s break down the specific loop in your code step by step, using a clear example to make it more understandable. 

### Code Explanation

The key part of the code is this loop:

```python
for i, app in enumerate(apps):
    df2 = dfa[dfa.Category == app]
    df3 = df2.head(5)
    plt.subplot(4, 2, i + 1)
    sns.barplot(data=df3, x='Installs', y='App')
    plt.xlabel('Installation in Millions')
    plt.ylabel('')
    plt.title(app, size=20)
```

### Step-by-Step Breakdown

1. **Enumerate the List of Apps**:
   ```python
   for i, app in enumerate(apps):
   ```
   - The `enumerate(apps)` function returns both the index (`i`) and the category name (`app`) from the list `apps`. 
   - For example, if `apps` is `['GAME', 'COMMUNICATION', 'PRODUCTIVITY', 'SOCIAL']`, the loop will iterate through:
     - **Iteration 1**: `i = 0`, `app = 'GAME'`
     - **Iteration 2**: `i = 1`, `app = 'COMMUNICATION'`
     - **Iteration 3**: `i = 2`, `app = 'PRODUCTIVITY'`
     - **Iteration 4**: `i = 3`, `app = 'SOCIAL'`

2. **Filter DataFrame for the Current Category**:
   ```python
   df2 = dfa[dfa.Category == app]
   ```
   - This line filters the `dfa` DataFrame to include only the rows where the 'Category' matches the current `app`.
   - **Example**: 
     - If we are on the first iteration where `app = 'GAME'`, `df2` will only contain the rows from `dfa` where the category is 'GAME'.

3. **Select the Top 5 Apps**:
   ```python
   df3 = df2.head(5)
   ```
   - This line selects the top 5 apps from the filtered DataFrame `df2`.
   - **Example**: 
     - Let's say the filtered DataFrame `df2` for 'GAME' looks like this:

     | Category | App               | Installs |
     |----------|-------------------|----------|
     | GAME     | Subway Surfers     | 1000000  |
     | GAME     | Candy Crush Saga   | 1500000  |
     | GAME     | PUBG Mobile        | 2000000  |
     | GAME     | Call of Duty       | 1800000  |
     | GAME     | Clash of Clans     | 1200000  |
     | GAME     | Temple Run         | 800000   |
     | ...      | ...                | ...      |

     The `df3` DataFrame will now contain:

     | Category | App               | Installs |
     |----------|-------------------|----------|
     | GAME     | PUBG Mobile        | 2000000  |
     | GAME     | Candy Crush Saga   | 1500000  |
     | GAME     | Call of Duty       | 1800000  |
     | GAME     | Subway Surfers     | 1000000  |
     | GAME     | Clash of Clans     | 1200000  |

4. **Create a Subplot for Each Category**:
   ```python
   plt.subplot(4, 2, i + 1)
   ```
   - This creates a subplot in a grid of 4 rows and 2 columns. The position of the current subplot is determined by `i + 1`, which means:
     - **Iteration 1** (`i=0`): Position (1, 1)
     - **Iteration 2** (`i=1`): Position (1, 2)
     - **Iteration 3** (`i=2`): Position (2, 1)
     - **Iteration 4** (`i=3`): Position (2, 2)

5. **Create a Bar Plot**:
   ```python
   sns.barplot(data=df3, x='Installs', y='App')
   ```
   - This line creates a bar plot using Seaborn for the top 5 apps in the current category (`df3`).
   - The x-axis will represent the number of installations, and the y-axis will represent the app names.

6. **Label the Axes and Title**:
   ```python
   plt.xlabel('Installation in Millions')
   plt.ylabel('')
   plt.title(app, size=20)
   ```
   - Here, the x-axis is labeled 'Installation in Millions', and the y-axis is left blank (because we are showing apps).
   - The title of the plot is set to the current category name (`app`), which helps to identify which category the subplot represents.

### Summary with an Example

Let’s say your `apps` list contains `['GAME', 'COMMUNICATION']`.

- **For `GAME`**:
  - `df2` filters for games.
  - `df3` contains the top 5 games, and a bar plot shows their installations.
  
- **For `COMMUNICATION`**:
  - `df2` filters for communication apps.
  - `df3` contains the top 5 communication apps, and a bar plot shows their installations.

Each subplot will clearly show the most installed apps in the respective category, allowing you to compare their popularity at a glance. 


### Code Explanation

1. **Grouping and Summing Ratings**:
   ```python
   rating = df_copy.groupby(['Category', 'Installs', 'App'])['Rating'].sum().sort_values(ascending=False).reset_index()
   ```
   - **Purpose**: This line groups the DataFrame `df_copy` by 'Category', 'Installs', and 'App', then sums the 'Rating' values for each group. Finally, it sorts the results in descending order.
   - **Example**:
     - Suppose your DataFrame `df_copy` contains the following data:

     | Category | Installs | App                        | Rating |
     |----------|----------|---------------------------|--------|
     | FAMILY   | 1000     | CS & IT Interview Questions| 5.0    |
     | GAME     | 2000     | Candy Crush Saga          | 4.8    |
     | GAME     | 1500     | Subway Surfers            | 5.0    |
     | COMMUNICATION | 500  | Hangouts                  | 4.9    |
     | FAMILY   | 800      | CT Brain Interpretation    | 5.0    |

     - After executing this line, `rating` will look like this:

     | Category | Installs | App                        | Rating |
     |----------|----------|---------------------------|--------|
     | FAMILY   | 1000     | CS & IT Interview Questions| 5.0    |
     | FAMILY   | 800      | CT Brain Interpretation    | 5.0    |
     | GAME     | 1500     | Subway Surfers            | 5.0    |
     | ...      | ...      | ...                       | ...    |

2. **Filtering for 5.0 Ratings**:
   ```python
   toprating_apps = rating[rating.Rating == 5.0]
   ```
   - **Purpose**: This line filters the `rating` DataFrame to include only those rows where the 'Rating' is exactly 5.0.
   - **Example**:
     - Continuing from the previous example, `toprating_apps` will now contain:

     | Category | Installs | App                        | Rating |
     |----------|----------|---------------------------|--------|
     | FAMILY   | 1000     | CS & IT Interview Questions| 5.0    |
     | FAMILY   | 800      | CT Brain Interpretation    | 5.0    |
     | GAME     | 1500     | Subway Surfers            | 5.0    |

3. **Counting the Number of 5-Rated Apps**:
   ```python
   print("Number of 5 rated apps", toprating_apps.shape[0])
   ```
   - **Purpose**: This line counts the number of rows in `toprating_apps` and prints the result.
   - `toprating_apps.shape[0]` gives the number of rows, which indicates how many apps have a rating of 5.0.
   - **Example Result**: If the count is 271, the output will be:
     ```
     Number of 5 rated apps 271
     ```

4. **Getting the Top Rated App**:
   ```python
   toprating_apps.head(1)
   ```
   - **Purpose**: This line retrieves the first row of the `toprating_apps` DataFrame, which represents the highest rated app.
   - **Example Output**:
     ```
     Category	Installs	App	                    Rating
     0	FAMILY	1000	    CS & IT Interview Questions	5.0
     ```
   - This means that the top-rated app with a 5.0 rating is **"CS & IT Interview Questions"** in the **Family** category.

### Summary with Example

In summary, the code you wrote does the following:

- It analyzes the `df_copy` DataFrame to find out how many apps have a perfect rating of 5.0.
- After processing, it identifies that there are **271 apps** with a 5.0 rating.
- The top app is **"CS & IT Interview Questions"** from the **Family** category, which had 1000 installs.
Yes, you can certainly loop through the `rating` DataFrame to count the number of apps that have a rating of 5.0. Here’s how you can do it using a simple loop, along with a more efficient approach using `pandas` methods. 

### Using a Loop

Here's how you can use a loop to count the number of apps with a 5.0 rating:

```python
# Initialize a counter
count_five_rated_apps = 0

# Loop through the 'Rating' column in the 'rating' DataFrame
for index, row in rating.iterrows():
    if row['Rating'] == 5.0:
        count_five_rated_apps += 1

# Print the result
print("Number of 5 rated apps:", count_five_rated_apps)
```

### Explanation

1. **Initialize a Counter**: Start with a counter (`count_five_rated_apps`) set to 0.
2. **Iterate Over Rows**: Use `iterrows()` to loop through each row of the `rating` DataFrame.
3. **Check Rating**: Inside the loop, check if the 'Rating' of the current row is 5.0. If it is, increment the counter by 1.
4. **Print the Result**: After the loop, print the total count of apps with a 5.0 rating.

### Using Pandas (More Efficient)

While the loop method works, it's generally more efficient to use built-in `pandas` functions. Here’s how to do it in a more concise way:

```python
# Count the number of apps with a rating of 5.0
count_five_rated_apps = rating[rating['Rating'] == 5.0].shape[0]

# Print the result
print("Number of 5 rated apps:", count_five_rated_apps)
```

### Explanation

1. **Filter the DataFrame**: `rating[rating['Rating'] == 5.0]` filters the DataFrame to keep only the rows where the rating is 5.0.
2. **Count the Rows**: `.shape[0]` gives the number of rows in the filtered DataFrame, which represents the count of apps with a 5.0 rating.
3. **Print the Result**: Finally, you print the total count.

### Conclusion

Using built-in methods is generally faster and more efficient than looping through rows, especially for large datasets. However, both methods achieve the same result. You can choose the one that you find more intuitive!
