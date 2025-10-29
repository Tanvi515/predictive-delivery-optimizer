import os
import pandas as pd
from typing import Dict, Optional

DATA_DIR_DEFAULT = "Case study internship data"

def _safe_read(path: str, parse_dates=None, **kwargs) -> Optional[pd.DataFrame]:
    """Read CSV if it exists, otherwise return None. Wrap pd.read_csv with errors caught."""
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path, parse_dates=parse_dates, **kwargs)
    except Exception as e:
        # If parsing with parse_dates fails, try without parse_dates as a fallback
        try:
            return pd.read_csv(path, **kwargs)
        except Exception:
            raise RuntimeError(f"Failed to read CSV {path}: {e}")

def load_all(data_dir: str = DATA_DIR_DEFAULT) -> Dict[str, Optional[pd.DataFrame]]:
    """
    Load all expected CSV files into a dictionary.
    Returns a dict with keys:
      'orders', 'delivery', 'routes', 'fleet', 'inventory', 'feedback', 'cost'
    Each value is a pandas.DataFrame or None if file not found.
    """
    files = {
        "orders": ("orders.csv", {"parse_dates": ["order_date"]}),
        # delivery contains promised and actual delivery timestamps
        "delivery": ("delivery_performance.csv", {"parse_dates": ["promised_time", "actual_delivery_time"]}),
        "routes": ("routes_distance.csv", {}),
        "fleet": ("vehicle_fleet.csv", {}),
        "inventory": ("warehouse_inventory.csv", {"parse_dates": ["last_restocked_date"]}),
        "feedback": ("customer_feedback.csv", {"parse_dates": ["feedback_date"]}),
        "cost": ("cost_breakdown.csv", {}),
    }

    out = {}
    for key, (fname, read_opts) in files.items():
        path = os.path.join(data_dir, fname)
        df = _safe_read(path, **read_opts)
        out[key] = df

    return out

if __name__ == "__main__":
    # quick local sanity check when running data.py directly
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else DATA_DIR_DEFAULT
    data = load_all(data_dir)
    for k, v in data.items():
        if v is None:
            print(f"[MISSING] {k}: expected file not found in '{data_dir}'")
        else:
            print(f"[OK] {k}: {v.shape[0]} rows x {v.shape[1]} cols — preview:")
            print(v.head(2).to_string(index=False))
            print("-" * 40)
