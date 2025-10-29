import pandas as pd
import numpy as np

def merge_datasets(d):
    orders = d.get('orders')
    delivery = d.get('delivery')
    routes = d.get('routes')
    cost = d.get('cost')

    df = orders.copy() if orders is not None else pd.DataFrame()
    if delivery is not None and 'Order_ID' in delivery.columns:
        df = df.merge(delivery, on='Order_ID', how='left')
    if routes is not None and 'Order_ID' in routes.columns:
        df = df.merge(routes, on='Order_ID', how='left')
    if cost is not None and 'Order_ID' in cost.columns:
        df = df.merge(cost, on='Order_ID', how='left')
    return df

def engineer_features(df):
    df = df.copy()

    # Standardize column names for consistency
    df.columns = [c.strip().lower() for c in df.columns]

    # Rename dataset-specific columns to expected ones
    rename_map = {
        'order_date': 'order_date',
        'promised_delivery_days': 'promised_days',
        'actual_delivery_days': 'actual_days',
        'delivery_cost_inr': 'delivery_cost',
        'distance_km': 'distance_km',
        'fuel_consumption_l': 'fuel_consumption',
        'traffic_delay_minutes': 'traffic_delay'
    }
    df.rename(columns=rename_map, inplace=True)

    if 'order_date' in df.columns:
        df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
        df['order_weekday'] = df['order_date'].dt.day_name()
        df['order_hour'] = df['order_date'].dt.hour

    if 'promised_days' in df.columns and 'actual_days' in df.columns:
        df['eta_gap_days'] = df['actual_days'] - df['promised_days']
        df['delay_flag'] = (df['eta_gap_days'] > 0).astype(int)
    else:
        df['delay_flag'] = np.nan

    for c in ['distance_km', 'fuel_consumption', 'traffic_delay', 'delivery_cost']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    return df
