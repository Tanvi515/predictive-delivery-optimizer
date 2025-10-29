# features.py - merging and feature engineering
import pandas as pd
import numpy as np

def merge_datasets(d):
    orders = d.get('orders')
    delivery = d.get('delivery')
    routes = d.get('routes')
    cost = d.get('cost')

    df = orders.copy() if orders is not None else pd.DataFrame()
    if delivery is not None and 'order_id' in delivery.columns:
        df = df.merge(delivery, on='order_id', how='left', suffixes=('','_del'))
    if routes is not None and 'order_id' in routes.columns:
        df = df.merge(routes, on='order_id', how='left')
    if cost is not None and 'order_id' in cost.columns:
        df = df.merge(cost, on='order_id', how='left')
    return df

def engineer_features(df):
    df = df.copy()
    for col in ['order_date','promised_time','actual_delivery_time']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    if 'order_date' in df.columns:
        df['order_weekday'] = df['order_date'].dt.day_name()
        df['order_hour'] = df['order_date'].dt.hour
    print("Columns in merged DataFrame:", df.columns.tolist())

    if 'promised_time' in df.columns and 'actual_delivery_time' in df.columns:
        df['eta_gap_minutes'] = (df['actual_delivery_time'] - df['promised_time']).dt.total_seconds()/60
        df['delay_flag'] = (df['eta_gap_minutes'] > 0).astype(int)
    for c in ['distance_km','fuel_consumption','traffic_delay','delivery_cost']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    return df
