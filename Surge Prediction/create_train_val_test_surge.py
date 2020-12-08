import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_surge_data(data_dir, month_nums):
    if month_nums == "all":
        data_paths = [data_dir + "/surge_2019-0" + str(i) + ".csv" for i in range(1, 7)]
        surge_DFs = [pd.read_csv(data_path) for data_path in data_paths]
        surge_DF = surge_DFs[0].append(surge_DFs[1:])
    else:
        data_path = data_dir + "/surge_2019-0" + str(month_nums) + ".csv"
        surge_DF = pd.read_csv(data_path)

    X = surge_DF[surge_DF.columns[2:]].values[:-1].astype(int)
    y = surge_DF[surge_DF.columns[9:]].values[1:]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=1)

    print("X train: ", X_train.shape, "y train: ", y_train.shape)
    print("X val: ", X_val.shape, "y val: ", y_val.shape)
    print("X test: ", X_test.shape, "y test: ", y_test.shape)

    return X_train, y_train, X_test, y_test, X_val, y_val

def main():
    load_surge_data("../data", 1)



if __name__ == '__main__':
    main()