import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


def read_data():
    print(f'Reading CSV file: insurance.csv into dataframe with Panda {pd.__version__}')
    # read a CSV file with a comma as a delimiter, assume no header, and skip the first column
    insurance_dataFrame = pd.read_csv("../data/insurance.csv", sep=',')
    print(f"Head after read:\n{insurance_dataFrame.head()}")
    print(f"Shape:\n{insurance_dataFrame.shape}")
    insurance_dataFrame["charges"].hist(bins=10)
    plt.show()


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
