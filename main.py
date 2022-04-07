import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

insurance_dataFrame = pd.read_csv("insurance.csv", sep=',')


def read_data():
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
    insurance_dataFrame["charges"].hist(bins=13, legend=True)
    plt.show()
    insurance_dataFrame["bmi"].hist(bins=13, legend=True)
    plt.show()
    insurance_dataFrame["children"].hist(bins=13, legend=True)
    plt.show()
    insurance_dataFrame["age"].hist(bins=13, legend=True)
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


def normalize(df):
    df_scaled = df.copy()
    for column in df_scaled.columns:
        if column == 1:
            df_scaled[column] = df_scaled[column]
        else:
            df_scaled[column] = (df_scaled[column] - df_scaled[column].min()) / (
                        df_scaled[column].max() - df_scaled[column].min())
    return df_scaled


def model_data():
    model = LinearRegression()


def linear_regression_simple_example():
    # You should call .reshape() on x because this array is required to be two-dimensional, or to be more precise,
    # to have one column and as many rows as necessary. That’s exactly what the argument (-1, 1) of .reshape() specifies.
    # Reshape gives a new shape to an array without changing its data.
    # Unspecified n rows (-1 means unspecified) and 1 column:
    x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
    y = np.array([5, 20, 14, 32, 22, 38])
    print(x)
    print(x.shape)
    print(y)
    print(x.shape)

    # Create an instance of the class LinearRegression, which will represent the regression model.
    # With .fit(), you calculate the optimal values of the weights 𝑏₀ and 𝑏₁, using the existing input and output (x and y) as the arguments.
    # In other words, .fit() fits the model. It returns self, which is the variable model itself.
    model = LinearRegression().fit(x, y)

    # You can obtain the coefficient of determination (𝑅²) with .score() called on model:
    print('Coefficient of determination (𝑅²):', model.score(x, y))
    # The attributes of model are .intercept_, which represents the coefficient, 𝑏₀ and .coef_, which represents 𝑏₁.
    # The value 𝑏₀ = 5.63 (approximately) illustrates that your model predicts the response 5.63 when 𝑥 is zero.
    # The value 𝑏₁ = 0.54 means that the predicted response rises by 0.54 when 𝑥 is increased by one.
    print('intercept:', model.intercept_)
    print('slope:', model.coef_)

    y_pred = model.predict(x)
    print('predicted response using simpler method 1:', y_pred, sep='\n')

    # Another way of doing this:
    # you multiply each element of x with model.coef_ and add model.intercept_ to the product.
    # The output here differs from the previous example only in dimensions.
    # The predicted response is now a two-dimensional array, while in the previous case, it had one dimension.
    y_pred = model.intercept_ + model.coef_ * x
    print('predicted response using method 2:', y_pred, sep='\n')

    # If you reduce the number of dimensions of x to one, these two approaches will yield the same result.
    # You can do this by replacing x with x.reshape(-1), x.flatten(), or x.ravel() when multiplying it with model.coef_.
    y_pred = model.intercept_ + model.coef_ * x.flatten()
    print('predicted response using method 2:', y_pred, sep='\n')

    # In practice, regression models are often applied for forecasts.
    # This means that you can use fitted models to calculate the outputs based on some other, new inputs.
    # Below predict() is applied to the new regressor x_new and yields the response y_new.
    # Uses arange() from numpy to generate an array with the elements from 0 (inclusive) to 5 (exclusive), that is 0, 1, 2, 3, and 4.
    x_new = np.arange(5).reshape((-1, 1))
    print(x_new)
    y_new = model.predict(x_new)
    print(y_new)


def multivariate_linear_regression_example():
    x = [[0, 1], [5, 1], [15, 2], [25, 5], [35, 11], [45, 15], [55, 34], [60, 35]]
    y = [4, 5, 20, 14, 32, 22, 38, 43]
    x, y = np.array(x), np.array(y)
    print(x)
    print(y)
    model = LinearRegression().fit(x, y)

    # Obtain the value of 𝑅² using .score() and the values of the estimators of regression coefficients with .intercept_ and .coef_.
    # Again, .intercept_ holds the bias 𝑏₀, while now .coef_ is an array containing 𝑏₁ and 𝑏₂ respectively.
    print('Coefficient of determination (𝑅²):', model.score(x, y))
    print('intercept:', model.intercept_)
    print('slope:', model.coef_)

    y_pred = model.predict(x)
    print('predicted response using simplest method:', y_pred, sep='\n')
    # In the above example, the intercept is approximately 5.52, and this is the value of the predicted response when 𝑥₁ = 𝑥₂ = 0.
    # The increase of 𝑥₁ by 1 yields the rise of the predicted response by 0.45. Similarly, when 𝑥₂ grows by 1, the response rises by 0.26.
    # Using the second method:
    y_pred = model.intercept_ + np.sum(model.coef_ * x, axis=1)
    print('predicted response using the formula method:', y_pred, sep='\n')


if __name__ == '__main__':
    read_data()
    model_data()
    linear_regression_simple_example()
    multivariate_linear_regression_example()
