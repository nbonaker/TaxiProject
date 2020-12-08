import numpy as np
import pandas as pd
import datetime
import holidays
from scipy import sparse


def adj_matrix_from_df(df):
    adj_matrix = np.zeros((266, 266))

    for index, data in df.iterrows():
        adj_matrix[index] = data[0]

    return adj_matrix


def is_weekend_or_holiday(date):
    weekend_days = [5, 6]
    us_holidays = holidays.US()
    date_day = date.weekday()
    stringified_date = str(date)
    if date_day in weekend_days:
        return True
    elif stringified_date in us_holidays:
        return True
    else:
        return False


def one_hot_time(date):
    obj_time = date.time()
    if obj_time < datetime.time(2, 0, 0) or obj_time > datetime.time(22, 0, 0):
        return [0, 0, 0, 0, 0, 1]
    elif obj_time >= datetime.time(2, 0, 0) and obj_time < datetime.time(6, 0, 0):
        return [1, 0, 0, 0, 0, 0]
    elif obj_time >= datetime.time(6, 0, 0) and obj_time < datetime.time(10, 0, 0):
        return [0, 1, 0, 0, 0, 0]
    elif obj_time >= datetime.time(10, 0, 0) and obj_time < datetime.time(14, 0, 0):
        return [0, 0, 1, 0, 0, 0]
    elif obj_time >= datetime.time(14, 0, 0) and obj_time < datetime.time(18, 0, 0):
        return [0, 0, 0, 1, 0, 0]
    else:
        return [0, 0, 0, 0, 1, 0]


def create_surge_data(data_dir, month_num):
    data_path = data_dir + "/yellow_tripdata_2019-0"+str(month_num)+".csv"
    full_DF = pd.read_csv(data_path)
    full_DF["tpep_pickup_datetime"] = pd.to_datetime(full_DF["tpep_pickup_datetime"], format='%Y-%m-%d %H:%M:%S',
                                                     errors='ignore')
    full_DF["tpep_dropoff_datetime"] = pd.to_datetime(full_DF["tpep_dropoff_datetime"], format='%Y-%m-%d %H:%M:%S',
                                                      errors='ignore')
    # restrict date range to remove weird parses
    full_DF = full_DF[(full_DF["tpep_pickup_datetime"] >= datetime.datetime(2019, month_num, 1, 0, 00, 0)) &
                       (full_DF["tpep_pickup_datetime"] < datetime.datetime(2019, month_num+1, 1, 0, 00, 0))]

    base_time = min(full_DF["tpep_pickup_datetime"])
    end_time = max(full_DF["tpep_pickup_datetime"])

    print("date range:", base_time, end_time)
    cur_interval_start = base_time


    surge_df = pd.DataFrame(columns=['interval_datetime', 'is_holiday', "PU_time_2AM", "PU_time_6AM", "PU_time_10AM",
                                     "PU_time_2PM", "PU_time_6PM", "PU_time_10PM"] + ["loc_"+str(i) for i in range(1, 267)])
    while cur_interval_start < end_time:
        interval_DF = full_DF[(full_DF["tpep_pickup_datetime"] >= cur_interval_start) &
                       (full_DF["tpep_pickup_datetime"] < cur_interval_start + datetime.timedelta(0, 10*60))]
        print(cur_interval_start, len(interval_DF))

        interval_DF = interval_DF.groupby(["PULocationID", "DOLocationID"]).agg(
            {'fare_amount': ['count']})
        interval_DF.columns = ["num_rides"]
        interval_DF.reset_index()

        surge_matrix = adj_matrix_from_df(interval_DF)

        out_degrees = np.sum(surge_matrix, axis=1)
        in_degrees = np.sum(surge_matrix, axis=0)
        surge_values = out_degrees - in_degrees


        surge_df.loc[len(surge_df)] = [cur_interval_start, is_weekend_or_holiday(cur_interval_start)]+one_hot_time(cur_interval_start)+surge_values.tolist()

        # increment interval by 10 minutes
        cur_interval_start = cur_interval_start + datetime.timedelta(0, 10*60)

    surge_df.to_csv('data/surge_2019-0'+str(month_num)+'.csv')


def create_surge_data_graph(data_dir, month_num):
    data_path = data_dir + "/yellow_tripdata_2019-0"+str(month_num)+".csv"
    full_DF = pd.read_csv(data_path)
    full_DF["tpep_pickup_datetime"] = pd.to_datetime(full_DF["tpep_pickup_datetime"], format='%Y-%m-%d %H:%M:%S',
                                                     errors='ignore')
    full_DF["tpep_dropoff_datetime"] = pd.to_datetime(full_DF["tpep_dropoff_datetime"], format='%Y-%m-%d %H:%M:%S',
                                                      errors='ignore')
    # restrict date range to remove weird parses
    full_DF = full_DF[(full_DF["tpep_pickup_datetime"] >= datetime.datetime(2019, month_num, 1, 0, 00, 0)) &
                       (full_DF["tpep_pickup_datetime"] < datetime.datetime(2019, month_num+1, 1, 0, 00, 0))]

    base_time = min(full_DF["tpep_pickup_datetime"])
    end_time = max(full_DF["tpep_pickup_datetime"])

    print("date range:", base_time, end_time)
    cur_interval_start = base_time


    surge_df = pd.DataFrame(columns=['interval_datetime', 'is_holiday', "PU_time_2AM", "PU_time_6AM", "PU_time_10AM",
                                     "PU_time_2PM", "PU_time_6PM", "PU_time_10PM"] + ["loc_"+str(i) for i in range(1, 267)])

    int_num = 1
    while cur_interval_start < end_time:
        interval_DF = full_DF[(full_DF["tpep_pickup_datetime"] >= cur_interval_start) &
                       (full_DF["tpep_pickup_datetime"] < cur_interval_start + datetime.timedelta(0, 10*60))]
        print(cur_interval_start, len(interval_DF))

        interval_DF = interval_DF.groupby(["PULocationID", "DOLocationID"]).agg(
            {'fare_amount': ['count']})
        interval_DF.columns = ["num_rides"]
        interval_DF.reset_index()

        surge_matrix = sparse.coo_matrix(adj_matrix_from_df(interval_DF))

        sparse.save_npz("surge_prediction_data/scipy_graphs/graph_0"+str(month_num)+"-"+str(int_num)+".npz", surge_matrix)

        int_num += 1
        cur_interval_start = cur_interval_start + datetime.timedelta(0, 10 * 60)


    # surge_df.to_csv('data/surge_graph_2019-0'+str(month_num)+'.csv')


def main():
    # for i in range(1, 7):
    #     create_surge_data("data", i)
    for i in range(3, 7):
        create_surge_data_graph("data", i)


if __name__ == '__main__':
    main()