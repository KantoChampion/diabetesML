# Machine learning project that generates a predictive model based on existing data sets


#import required packages
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model, __all__
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import cluster, datasets, metrics, model_selection, linear_model

#Load the dataset
df = pd.read_csv('diabetes.csv')

def replaceNull():
    for column in df.columns:
        if df[column].isnull().sum() > 0:
            # Check the data type of the column, making sure its numeric
            if pd.api.types.is_numeric_dtype(df[column]):
                # Calculate the mean of the column, skipping NaN values
                column_mean = df[column].mean(skipna=True)
                # Replace NaN values with the calculated mean
                df[column] = df[column].fillna(column_mean)
            else:
                # For non-numeric columns, fill NaN values with the mode (most frequent value)
                mode_value = df[column].mode().iloc[0]
                df[column] = df[column].fillna(mode_value)


def trainData():
    x_train, x_test, y_train, y_test = train_test_split(df[['weight']], df[['glyhb']], test_size=0.2)
    reg = linear_model.LinearRegression().fit(x_train, y_train)
    print("Trained Data Score", reg.score(x_train, y_train))
    print("Test Score", reg.score(x_test, y_test))

    y_pred = reg.predict(x_test)
    #metrics.accuracy_score(y_test, y_pred)

    plt.scatter(x_train, y_train, color='blue')
    plt.xlabel('weight')
    plt.ylabel('glyhb')
    plt.title('Linear Regression')
    plt.show()

def main():
    print("Running Diabetes ML model")
    print(df.head())

    # Do some data wrangling and data processing to remove null values
    print(df.notnull())
    replaceNull()

    print(df.head())

    print(df.all)

    #Creating plots
    sns.pairplot(df, x_vars="chol", y_vars="waist", hue="waist")
    plt.show()

    sns.lmplot(x="chol", y="waist", data=df)
    plt.show()

    sns.jointplot(x="waist", y="chol", data=df)
    plt.show()

    sns.pairplot(df, x_vars="hdl", y_vars="bp.2d", hue="waist")
    plt.show()

    #Train the dataset and predict glyhb levels based on data
    trainData()


if __name__ == "__main__":
    main()