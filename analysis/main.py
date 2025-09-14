import polars as pl
from fitparse import FitFile

import pathlib
import os

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from sklearn.neighbors import NearestNeighbors

this_dir = pathlib.Path('.').parent.resolve()
data_dir = this_dir / 'data'

def process_fit(file_path):
    file = FitFile(str(file_path))
    data = []
    for record in file.get_messages('record'):
        record_data = {}
        for data_field in record:
            record_data[data_field.name] = data_field.value
        data.append(record_data)
    df = pl.DataFrame(data)
    return df

if __name__ == "__main__":
    workouts = [data_dir / file for file in os.listdir(data_dir) if file.endswith('.fit')]
    workouts.sort(reverse=True)
    all_data = []
    for workout in workouts:
        print(f"Processing {workout.name}")
        if 'Cycling' not in workout.name:
            continue
        df = process_fit(workout)
        df = df.with_columns(pl.lit(workout.name).alias('workout_name'))
        all_data.append(df.to_pandas())

    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df = pl.from_pandas(combined_df)
    combined_df = combined_df.drop_nulls(subset=['position_lat', 'position_long', 'altitude'])

    print(combined_df)

    distances = NearestNeighbors(n_neighbors=25).fit(combined_df.select(['position_long', 'position_lat']).to_numpy())
    distances, _ = distances.kneighbors(combined_df.select(['position_long', 'position_lat']).to_numpy())
    print(distances.shape)
    combined_df.write_parquet(data_dir / 'combined.parquet')



    # fig = px.box(x=distances)
    # fig.show(renderer="browser")

    # fig = px.scatter(combined_df, x='position_long', y='position_lat', color='altitude',)
    # fig.update_layout(
    # scene=dict(aspectmode='manual', aspectratio=dict(x=1, y=1, z=0.1)))
    # fig.show(renderer="browser")
