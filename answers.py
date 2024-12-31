import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

try:
    # Load the Iris dataset
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

    print("Dataset loaded successfully!")

    # Task 1: Load and Explore the Dataset
    print("\nFirst 5 rows of the dataset:")
    print(df.head())

    print("\nDataset Info:")
    df.info()  # Check data types and non-null values

    # Check for missing values
    print("\nMissing Values:")
    print(df.isnull().sum())

    # No missing values in Iris dataset, but demonstrating the code
    # df.dropna(inplace=True) # Drops rows with missing values
    # df.fillna(df.mean(), inplace=True) # Fills missing values with the mean

    # Task 2: Basic Data Analysis
    print("\nDescriptive Statistics:")
    print(df.describe())

    print("\nMean of sepal length per species:")
    print(df.groupby('species')['sepal length (cm)'].mean())

    # Interesting finding: Setosa has the smallest sepal length on average

    # Task 3: Data Visualization
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']])
    plt.title('Feature Trends')
    plt.xlabel('Data Point Index')
    plt.ylabel('Value (cm)')
    plt.legend(title='Features')
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.barplot(x='species', y='petal length (cm)', data=df)
    plt.title('Average Petal Length per Species')
    plt.xlabel('Species')
    plt.ylabel('Petal Length (cm)')
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.histplot(df['sepal length (cm)'], kde=True)
    plt.title('Distribution of Sepal Length')
    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Frequency')
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', data=df)
    plt.title('Sepal Length vs. Petal Length')
    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Petal Length (cm)')
    plt.legend(title='Species')
    plt.show()

except FileNotFoundError:
    print("Error: Dataset file not found.")
except pd.errors.ParserError:
    print("Error: Could not parse the dataset file. Check the format.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")