import torch as th
import dgl
import os
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import sparse
import pickle

datetime.strptime('2019-01-02', '%Y-%m-%d')

time_bins = [
    'PU_time_2AM-5:59AM',
    'PU_time_6AM-9:59AM',
    'PU_time_10AM-1:59PM',
    'PU_time_2PM-5:59PM',
    'PU_time_6PM-9:59PM',
    'PU_time_10PM-1:59AM'
]

features = [
    'PU_time_2AM-5:59AM',
    'PU_time_6AM-9:59AM',
    'PU_time_10AM-1:59PM',
    'PU_time_2PM-5:59PM',
    'PU_time_6PM-9:59PM',
    'PU_time_10PM-1:59AM',
    'weekend/holiday',
    'PU_longitude',
    'PU_latitude',
    'DO_longitude',
    'DO_latitude',
    'distance'
]


def adj_matrix_from_df(df):
    adj_matrix = np.zeros((266, 266))

    for index, data in df.iterrows():
        adj_matrix[index] = data[0]

    return adj_matrix


def min_max_scalar(x):
    return (x - x.min()) / (x.max() - x.min())

generate_graphs = False
if generate_graphs:
    agg_fare_df = pd.read_csv('./fare_prediction_data/avg_fare_all.csv')
    agg_fare_df.drop(['Unnamed: 0'], axis=1, inplace=True)

    # Normalize features using min-max scalar
    agg_fare_df[features] = agg_fare_df[features].apply(min_max_scalar)

    # Convert Y-m-d strings to datetime objects
    agg_fare_df['pickup_date'] = agg_fare_df['pickup_date'].map(
        lambda datestring: datetime.strptime(datestring, '%Y-%m-%d'))
    # remove bad pickup date entries
    agg_fare_df = agg_fare_df[(agg_fare_df["pickup_date"] >= datetime(2019, 1, 1, 0, 00, 0)) &
                            (agg_fare_df["pickup_date"] < datetime(2019, 7, 1, 0, 00, 0))]

    unique_date_times = agg_fare_df.groupby(
        ['pickup_date'] + time_bins).size().reset_index().rename(columns={0: 'count'})
    print("Number of Graphs Total for Fare Prediction: ", len(unique_date_times))

    # Here is where we construct the graph for the graph (x, y) pairs
    # x is a graph consisting of location nodes representing pickup & dropoff locations in NYC
    # the edges have features such as the pickup, dropoff location, haversine distance between nodes,
    # whether the aggregate of rides occurs on weekend/holiday, and the 4-hour time-bin the rides occured in.
    # y is the target representing a graph where the edges have Avg Fare as the label.
    #
    # We calculate the MSE.
    # print(agg_fare_df.head(10))

    for index, row in unique_date_times.iterrows():
        # each entry in this dataframe is a directed edge in the graph (pickup node -> dropoff node) representing the aggregate
        # of all trips that occured between the pickup and dropoff nodes during a specific date and time interval.
        graph_df = agg_fare_df[(agg_fare_df['pickup_date'] == row['pickup_date']) &
                            (agg_fare_df['PU_time_2AM-5:59AM'] == row['PU_time_2AM-5:59AM']) &
                            (agg_fare_df['PU_time_6AM-9:59AM'] == row['PU_time_6AM-9:59AM']) &
                            (agg_fare_df['PU_time_10AM-1:59PM'] == row['PU_time_10AM-1:59PM']) &
                            (agg_fare_df['PU_time_2PM-5:59PM'] == row['PU_time_2PM-5:59PM']) &
                            (agg_fare_df['PU_time_6PM-9:59PM'] == row['PU_time_6PM-9:59PM']) &
                            (agg_fare_df['PU_time_10PM-1:59AM'] == row['PU_time_10PM-1:59AM'])]
        graph_df.to_csv("./fare_prediction_data/graph_tiny.csv")

        edges_start = graph_df["pickup_location_ID"].values.astype(int)
        edges_end = graph_df["dropoff_location_ID"].values.astype(int)

        edge_labels = graph_df["avg_fare"].values.astype(float)
        edge_features = graph_df[features].values.astype(float)

        data_dict = {"edges_start": edges_start, "edges_end": edges_end, "edge_labels": edge_labels, "edge_features": edge_features}
        pickle.dump(data_dict, open("./fare_prediction_data/pickles/graph_"+str(index)+".p", "wb"))
        print(index)

for filename in os.listdir("./fare_prediction_data/pickles/"):
    graph_data = pickle.load(open("./fare_prediction_data/pickles/"+filename, "rb"))
    edges_start = th.tensor(graph_data["edges_start"])
    edges_end = th.tensor(graph_data["edges_end"])

    edge_labels = th.tensor(graph_data["edges_start"])
    edge_features = th.tensor(graph_data["edges_start"])

    g = dgl.graph((edges_start, edges_end))
    g.edata["label"] = edge_labels
    g.edata["feature"] = edge_features
    print(g)
    break
