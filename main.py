import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression


def read_data():
    print(f'Reading CSV file: insurance.csv into dataframe with Panda {pd.__version__}')
    # read a CSV file with a comma as a delimiter, assume no header, and skip the first column
    insurance_dataFrame = pd.read_csv("insurance.csv", sep=',')

    print(f"Head after read:\n{insurance_dataFrame.head()}")
    print(f"Shape:\n{insurance_dataFrame.shape}")
    print(f"Describe the dataframe: \n{insurance_dataFrame.describe()}")
    print(f"Describe the charges column:\n{insurance_dataFrame['charges'].describe()}")

    # group by diagnosis - this returns groupby object.
    regions_groupby = insurance_dataFrame.groupby("region")
    # Use size to get the count
    print(f"Count of regions:\n {regions_groupby.size()}")
    # Another way to get a count per region:
    print(f"Another way to get a count per region => insurance_dataFrame['region'].value_counts() =\n {insurance_dataFrame['region'].value_counts()}")

    # Display histogram of charges
    # Because the mean value is higher than the median value, this implies that the distribution of the insurance charges is skewed to the higher charge value (right skewed).
    insurance_dataFrame["charges"].hist(bins=13)
    plt.show()
    insurance_dataFrame["bmi"].hist(bins=13)
    plt.show()
    insurance_dataFrame["children"].hist(bins=13)
    plt.show()
    insurance_dataFrame["age"].hist(bins=13)
    plt.show()

    correlation_matrix = insurance_dataFrame.corr()
    print(f"Correlation Matrix Shape:\n{correlation_matrix}")
    # Age and BMI have a weak positive correlation, meaning that as someone ages, their body mass tends to increase.
    # There is also a moderate positive correlation between age and charges, BMI and charges, and children and expenses.
    # These associations imply that as age, BMI, and the number of children increase, the expected cost of insurance goes up.

    plt.scatter(insurance_dataFrame.age, insurance_dataFrame.charges)
    plt.show()
    plt.scatter(insurance_dataFrame.bmi, insurance_dataFrame.charges)
    plt.show()
    plt.scatter(insurance_dataFrame.children, insurance_dataFrame.charges)
    plt.show()

    # x = []
    # y = []
    # model = LinearRegression()
    # model = LinearRegression().fit(x, y)


def normalize(df):
    df_scaled = df.copy()
    for column in df_scaled.columns:
        if column == 1:
            df_scaled[column] = df_scaled[column]
        else:
            df_scaled[column] = (df_scaled[column] - df_scaled[column].min()) / (
                        df_scaled[column].max() - df_scaled[column].min())
    return df_scaled


if __name__ == '__main__':
    read_data()
