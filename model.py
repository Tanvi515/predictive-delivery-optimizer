import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

MODEL_PATH = "delay_model.pkl"

def select_features(df):
    features = [c for c in ['distance_km','fuel_consumption','traffic_delay','order_hour','delivery_cost'] if c in df.columns]
    return features

def train_model(df):
    X = df[select_features(df)].fillna(0)
    y = df['delay_flag'].fillna(0).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    joblib.dump(clf, MODEL_PATH)
    preds = clf.predict(X_test)
    return {"accuracy": float(accuracy_score(y_test, preds))}

def load_model(path=MODEL_PATH):
    return joblib.load(path) if os.path.exists(path) else None
