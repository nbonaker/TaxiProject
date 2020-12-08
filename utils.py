import argparse
import pandas as pd
import numpy as np
import datetime
import holidays
import networkx as nx
from matplotlib import pyplot as plt

time_of_day_bins = ['PU_time_2AM-5:59AM', 'PU_time_6AM-9:59AM', 'PU_time_10AM-1:59PM',
                    'PU_time_2PM-5:59PM', 'PU_time_6PM-9:59PM', 'PU_time_10PM-1:59AM']
weekend_days = [5, 6]
us_holidays = holidays.US()

arranged_cols = ['pickup_location_ID', 'dropoff_location_ID',
    'pickup_date', 'avg_fare', 'PU_time_2AM-5:59AM',
    'PU_time_6AM-9:59AM', 'PU_time_10AM-1:59PM',
    'PU_time_2PM-5:59PM', 'PU_time_6PM-9:59PM',
    'PU_time_10PM-1:59AM', 'weekend/holiday',
    'PU_longitude','PU_latitude', 'DO_longitude',
    'DO_latitude', 'distance']


def create_trip_DF(path):
    df = pd.read_csv(path)
    df.drop(['extra', 'RatecodeID', 'store_and_fwd_flag', 'RatecodeID', 'payment_type', 'passenger_count', 'store_and_fwd_flag', 'tolls_amount',
             'VendorID', 'tip_amount', 'improvement_surcharge', 'total_amount', 'congestion_surcharge', 'mta_tax'], axis=1, inplace=True)
    df.rename(columns={'tpep_pickup_datetime': 'pickup_datetime', 'tpep_dropoff_datetime': 'dropoff_datetime',
                       'PULocationID': 'pickup_location_ID', 'DOLocationID': 'dropoff_location_ID'}, inplace=True)
    df["pickup_datetime"] = pd.to_datetime(
        df["pickup_datetime"], format='%Y-%m-%d %H:%M:%S', errors='ignore')
    df["dropoff_datetime"] = pd.to_datetime(
        df["dropoff_datetime"], format='%Y-%m-%d %H:%M:%S', errors='ignore')
    df.dropna()
    df = df[(df['fare_amount'] >= 2.50) & (
        df['fare_amount'] <= 150) & (df['trip_distance'] > 0)]
    return df


def get_time_of_day_bin(timestamp, time_of_day_bins):
    if 2 <= timestamp.hour and timestamp.hour < 6:
        return time_of_day_bins[0]
    elif 6 <= timestamp.hour and timestamp.hour < 10:
        return time_of_day_bins[1]
    elif 10 <= timestamp.hour and timestamp.hour < 14:
        return time_of_day_bins[2]
    elif 14 <= timestamp.hour and timestamp.hour < 18:
        return time_of_day_bins[3]
    elif 18 <= timestamp.hour and timestamp.hour < 22:
        return time_of_day_bins[4]
    else:
        return time_of_day_bins[5]


def append_and_fill_time_of_day_bins(df, time_of_day_bins):
    for time_bin in time_of_day_bins:
        df[time_bin] = df['pickup_datetime'].map(lambda timestamp: 1 if get_time_of_day_bin(
            timestamp, time_of_day_bins) == time_bin else 0)
    return df


def create_avg_fare_df(df):
    fare_prediction_df = df.groupby(["pickup_location_ID", "dropoff_location_ID", "pickup_date"] +
                                    time_of_day_bins, as_index=False).agg(avg_fare=('fare_amount', 'mean'))
    return fare_prediction_df


def is_weekend_or_holiday(date):
    date_day = date.day
    stringified_date = str(date)
    if date_day in weekend_days:
        return True
    elif stringified_date in us_holidays:
        return True
    else:
        return False


def calculate_haversine_distance(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    lon_diff = lon2 - lon1
    lat_diff = lat2 - lat1
    a = np.sin(lat_diff/2.0)**2 + np.cos(lat1) * \
        np.cos(lat2) * np.sin(lon_diff/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    earth_radius_mi = 3956  # radius of Earth in miles
    mi = earth_radius_mi * c
    return mi


def add_PU_DO_coordinates(df, zone_lookup_path):
    zone_lat_long_lookup = pd.read_csv(zone_lookup_path)[
        ['LocationID', 'X', 'Y']]
    # add coordinates for pickup location
    df = df.merge(zone_lat_long_lookup, how='left',
                  left_on='pickup_location_ID', right_on='LocationID')
    df.dropna(inplace=True)
    df.rename(columns={
        'X': 'PU_longitude',
        'Y': 'PU_latitude'
    }, inplace=True)
    del df['LocationID']
    # add coordinates for dropoff location
    df = df.merge(zone_lat_long_lookup, how='left',
                  left_on='dropoff_location_ID',                      right_on='LocationID')
    df.dropna(inplace=True)
    df.rename(columns={
        'X': 'DO_longitude',
        'Y': 'DO_latitude'
    }, inplace=True)
    del df['LocationID']


def add_haversine_distance(df):
    df['distance'] = df.apply(
        lambda row: calculate_haversine_distance(
            row['PU_longitude'],
            row['PU_latitude'],
            row['DO_longitude'],
            row['DO_latitude']
        ), axis=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, default="data/")
    parser.add_argument('--filename', type=str,
                        default='yellow_tripdata_2019-01.csv')
    args = parser.parse_args()

    # load dataframe and drop useless columns
    path = args.folder + args.filename
    trip_DF = create_trip_DF(path)
    # add one-hot time bins
    append_and_fill_time_of_day_bins(trip_DF, time_of_day_bins)
    # transform dates
    trip_DF['pickup_date'] = trip_DF['pickup_datetime'].map(
        lambda timestamp: timestamp.date())

    # create fare prediction dataset
    trip_avg_fare_DF = create_avg_fare_df(trip_DF)
    # if trips occur on weekend or US holiday
    trip_avg_fare_DF['weekend/holiday'] = trip_avg_fare_DF['pickup_date'].map(
        lambda date: 1 if is_weekend_or_holiday(date) else 0)
    # Cast pickup and dropoff IDs to ints
    trip_avg_fare_DF = trip_avg_fare_DF.astype(
        {"pickup_location_ID": int, "dropoff_location_ID": int})
    # Add pickup and dropoff coordinates to df
    add_PU_DO_coordinates(trip_avg_fare_DF, './taxi_zones.csv')
    # Add haversine distance between pickup and dropoff coordinates
    add_haversine_distance(trip_avg_fare_DF)
    # Need to drop the following columns when doing training & testing
    # 1. pickup_location_ID
    # 2. dropoff_location_ID
    # 3. pickup_date
    # 4. avg_fare as this is the target

    # Rearrange cols
    trip_avg_fare_DF = trip_avg_fare_DF[arranged_cols]

    trip_avg_fare_DF.to_csv(
        'preprocessed_fare_prediction/processed_' + args.filename)
