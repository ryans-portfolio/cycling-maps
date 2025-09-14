import polars as pl
from fitparse import FitFile

import pathlib
import os

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

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
        df = process_fit(workout)
        df = df.with_columns(pl.lit(workout.name).alias('workout_name'))
        all_data.append(df.to_pandas())

    combined_df = pd.concat(all_data, ignore_index=True)    
    print(combined_df)

    fig = px.scatter_3d(combined_df, x='position_lat', y='position_long', z='altitude',)
    fig.update_layout(
    scene=dict(aspectmode='manual', aspectratio=dict(x=1, y=1, z=0.1)))
    fig.show(renderer="browser")
