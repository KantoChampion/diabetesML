
#import required packages
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import linear_model
from sklearn.model_selection import train_test_split

#Load the dataset
df = pd.read_csv('housing.csv')

def replaceNull():
    for column in df.columns:
        if df[column].isnull().sum() > 0:
            # Check the data type of the column
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
    x_train, x_test, y_train, y_test = train_test_split(df[['price']], df[['lotsize']], test_size=0.2)
    reg = linear_model.LinearRegression().fit(x_train, y_train)
    print("Trained Data Score", reg.score(x_train, y_train))
    print("Test Score", reg.score(x_test, y_test))

    y_pred = reg.predict(x_test)
    #metrics.accuracy_score(y_test, y_pred)

    plt.scatter(x_train, y_train, color='blue')
    plt.xlabel('price')
    plt.ylabel('lotsize')
    plt.title('Linear Regression')
    plt.show()

def main():
    print("Running Housing ML model")
    print(df.head())

    # Do some data wrangling and data processing to remove null values
    print(df.notnull())
    replaceNull()

    print(df.head())

    print(df.all)

    #Creating plots
    sns.pairplot(df, x_vars="lotsize", y_vars="price", hue="price")
    plt.show()

    #Train the dataset and predict glyhb levels based on data
    trainData()


if __name__ == "__main__":
    main()