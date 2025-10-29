def compute_kpis(df):
    out = {}
    out['total_orders'] = int(df.shape[0])
    out['on_time_pct'] = float(100*(1 - df['delay_flag'].mean())) if 'delay_flag' in df.columns else None
    out['avg_delivery_cost'] = float(df['delivery_cost'].mean()) if 'delivery_cost' in df.columns else None
    out['avg_distance_km'] = float(df['distance_km'].mean()) if 'distance_km' in df.columns else None
    return out
