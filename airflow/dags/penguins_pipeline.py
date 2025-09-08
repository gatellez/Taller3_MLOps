
from __future__ import annotations
import os
from datetime import datetime

import pandas as pd
import pymysql
from airflow import DAG
from airflow.operators.python import PythonOperator
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

DATA_DB_HOST = os.environ.get("DATA_DB_HOST", "penguins-db")
DATA_DB_PORT = int(os.environ.get("DATA_DB_PORT", "3306"))
DATA_DB_USER = os.environ.get("DATA_DB_USER", "penguins")
DATA_DB_PASSWORD = os.environ.get("DATA_DB_PASSWORD", "penguins")
DATA_DB_NAME = os.environ.get("DATA_DB_NAME", "penguins")

RAW_TABLE = "penguins_raw"
PREP_TABLE = "penguins_prepared"
ARTIFACT_DIR = "/opt/airflow/models"
MODEL_PATH = os.path.join(ARTIFACT_DIR, "model.pkl")

def get_conn():
    return pymysql.connect(
        host=DATA_DB_HOST,
        port=DATA_DB_PORT,
        user=DATA_DB_USER,
        password=DATA_DB_PASSWORD,
        database=DATA_DB_NAME,
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True,
    )

def fetch_df(query: str, params=None) -> pd.DataFrame:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(query, params or ())
        rows = cur.fetchall()
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows)

def clear_database_tables(**_):
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {PREP_TABLE}")
        cur.execute(f"DROP TABLE IF EXISTS {RAW_TABLE}")

def load_raw_penguins(**_):
    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"
    df = pd.read_csv(url)
    df = df.astype(object).where(~pd.isna(df), None)

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {RAW_TABLE}")
        cur.execute(f"""
            CREATE TABLE {RAW_TABLE} (
                species VARCHAR(50) NULL,
                island VARCHAR(50) NULL,
                bill_length_mm FLOAT NULL,
                bill_depth_mm FLOAT NULL,
                flipper_length_mm FLOAT NULL,
                body_mass_g FLOAT NULL,
                sex VARCHAR(20) NULL
            )
        """)
        rows = list(df[[
            "species","island","bill_length_mm","bill_depth_mm","flipper_length_mm","body_mass_g","sex"
        ]].itertuples(index=False, name=None))
        if rows:
            sql = f"""
                INSERT INTO {RAW_TABLE}
                (species,island,bill_length_mm,bill_depth_mm,flipper_length_mm,body_mass_g,sex)
                VALUES (%s,%s,%s,%s,%s,%s,%s)
            """
            cur.executemany(sql, rows)

def preprocess(**_):
    df = fetch_df(f"SELECT * FROM {RAW_TABLE}")
    df = df.dropna(subset=["species"]).copy()

    X = df.drop(columns=["species"])
    y = df["species"].astype(str)
    Xy = pd.concat([X, y.rename("species")], axis=1)
    Xy = Xy.astype(object).where(~pd.isna(Xy), None)

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {PREP_TABLE}")
        cur.execute(f"""
            CREATE TABLE {PREP_TABLE} (
                island VARCHAR(50) NULL,
                bill_length_mm FLOAT NULL,
                bill_depth_mm FLOAT NULL,
                flipper_length_mm FLOAT NULL,
                body_mass_g FLOAT NULL,
                sex VARCHAR(20) NULL,
                species VARCHAR(50) NOT NULL
            )
        """)
        rows = list(Xy[[
            "island","bill_length_mm","bill_depth_mm","flipper_length_mm","body_mass_g","sex","species"
        ]].itertuples(index=False, name=None))
        if rows:
            sql = f"""
                INSERT INTO {PREP_TABLE}
                (island,bill_length_mm,bill_depth_mm,flipper_length_mm,body_mass_g,sex,species)
                VALUES (%s,%s,%s,%s,%s,%s,%s)
            """
            cur.executemany(sql, rows)

def train_model(**_):
    data = fetch_df(f"SELECT * FROM {PREP_TABLE}")
    if data.empty:
        raise RuntimeError("No hay datos en tabla preparada para entrenar.")

    y = data["species"].astype(str)
    X = data.drop(columns=["species"]).copy()

    numeric_cols = X.select_dtypes(include=["float64", "int64", "float32", "int32"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    preproc = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )
    pipe = Pipeline(steps=[
        ("preprocessor", preproc),
        ("model", RandomForestClassifier(n_estimators=200, random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipe.fit(X_train, y_train)
    acc = accuracy_score(y_test, pipe.predict(X_test))

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    joblib.dump({"pipeline": pipe, "accuracy": float(acc)}, MODEL_PATH)
    print(f"Modelo guardado en {MODEL_PATH} con accuracy={acc:.4f}")

with DAG(
    dag_id="penguins_etl_train",
    description="ETL + Preprocesamiento + Entrenamiento + Artefacto (PyMySQL, sin SQLAlchemy)",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["penguins","ml","mysql","pymysql"],
) as dag:
    t_clear = PythonOperator(task_id="clear_database_tables", python_callable=clear_database_tables)
    t_load  = PythonOperator(task_id="load_raw_penguins", python_callable=load_raw_penguins)
    t_pre   = PythonOperator(task_id="preprocess", python_callable=preprocess)
    t_train = PythonOperator(task_id="train_model", python_callable=train_model)

    t_clear >> t_load >> t_pre >> t_train
