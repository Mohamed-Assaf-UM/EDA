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
