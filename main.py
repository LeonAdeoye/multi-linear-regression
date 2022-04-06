import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression


def read_data():
    print(f'Reading CSV file: insurance.csv into dataframe with Panda {pd.__version__}')
    # read a CSV file with a comma as a delimiter, assume no header, and skip the first column
    insurance_dataFrame = pd.read_csv("../data/insurance.csv", sep=',')
    print(f"Head after read:\n{insurance_dataFrame.head()}")
    print(f"Shape:\n{insurance_dataFrame.shape}")

    # group by diagnosis - this returns groupby object.
    regions_groupby = insurance_dataFrame.groupby(by=["region"])
    # Use size to get the count
    print(f"Count of regions:\n {regions_groupby.size()}")

    correlation_matrix = insurance_dataFrame.corr()
    print(f"Shape:\n{correlation_matrix}")

    # Display histogram of charges
    insurance_dataFrame["charges"].hist(bins=13)
    plt.show()

    x = []
    y = []
    model = LinearRegression()
    model = LinearRegression().fit(x, y)


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
