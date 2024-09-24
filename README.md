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

