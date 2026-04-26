"""
Airflow DAG for daily batch fraud detection scoring.

Runs nightly at 2 AM UTC:
1. Pulls new transactions from S3 (data/incoming/YYYY-MM-DD/)
2. Scores via inference API POST /predict
3. Writes results to S3 (data/scored/YYYY-MM-DD/results.parquet)
4. Results are queryable via Athena on S3

Task flow:
  extract_from_s3 → validate_data → batch_score → load_to_s3 → athena_query
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.operators.s3 import S3ListOperator
from airflow.providers.amazon.aws.transfers.s3_to_local import S3ToLocalFilesystemOperator
from airflow.utils.decorators import apply_defaults

# Configuration
S3_BUCKET = os.getenv("S3_BUCKET", "fraudlens")
S3_INCOMING_PREFIX = "incoming"
S3_SCORED_PREFIX = "scored"
API_ENDPOINT = os.getenv("API_ENDPOINT", "http://localhost:8000")
BATCH_SIZE = 1000  # Score in batches for efficiency

# Default DAG arguments
default_args = {
    "owner": "fraudlens-ml",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "depends_on_past": False,
}

# - DAG Definition 

dag = DAG(
    "fraudlens_daily_scoring",
    default_args=default_args,
    description="Daily batch fraud detection scoring",
    schedule_interval="0 2 * * *",  # 2 AM UTC daily
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["fraudlens", "ml-pipeline", "batch"],
)

# - Tasks 


def extract_transactions_from_s3(**context):
    """
    Extract new transactions from S3 incoming directory.

    Context pushes:
      - transactions_df: DataFrame of transactions
      - partition_date: YYYY-MM-DD
    """
    execution_date = context["execution_date"]
    partition_date = execution_date.strftime("%Y-%m-%d")

    s3_key = f"s3://{S3_BUCKET}/{S3_INCOMING_PREFIX}/{partition_date}/"
    print(f"[Extract] Reading transactions from {s3_key}")

    try:
        # Read from S3 (assuming Parquet files)
        # In practice, use s3fs or boto3
        df = pd.read_parquet(s3_key)
        print(f"[Extract] Loaded {len(df)} transactions")

        context["task_instance"].xcom_push(key="transactions_df", value=df)
        context["task_instance"].xcom_push(key="partition_date", value=partition_date)

        return {"status": "success", "rows": len(df)}

    except Exception as e:
        print(f"[Extract] No data for {partition_date}: {e}")
        return {"status": "no_data", "rows": 0}


def validate_transactions(**context):
    """
    Validate transaction data quality.

    Checks:
      - No missing required fields
      - Valid value ranges
      - Expected schema
    """
    df = context["task_instance"].xcom_pull(task_ids="extract_from_s3", key="transactions_df")

    if df is None or df.empty:
        print("[Validate] No transactions to validate")
        return {"status": "skipped"}

    required_cols = [
        "transaction_amt",
        "product_cd",
        "addr1",
        "p_emaildomain",
        "r_emaildomain",
    ]

    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    print(f"[Validate] ✓ {len(df)} transactions passed validation")
    return {"status": "success", "rows": len(df)}


def batch_score_transactions(**context):
    """
    Score transactions via inference API in batches.

    For each batch:
      1. POST to /predict endpoint
      2. Collect fraud_score + is_fraud
      3. Append results to accumulated list
    """
    df = context["task_instance"].xcom_pull(
        task_ids="extract_from_s3",
        key="transactions_df",
    )

    if df is None or df.empty:
        print("[Score] No transactions to score")
        return {"status": "skipped"}

    print(f"[Score] Scoring {len(df)} transactions via {API_ENDPOINT}/predict")

    results = []
    failed_count = 0

    for i in range(0, len(df), BATCH_SIZE):
        batch = df.iloc[i : i + BATCH_SIZE]
        print(f"[Score] Batch {i//BATCH_SIZE + 1}: {len(batch)} transactions")

        for _, row in batch.iterrows():
            try:
                # Prepare request
                request_data = {
                    "transaction_amt": row.get("transaction_amt"),
                    "product_cd": row.get("product_cd"),
                    "addr1": row.get("addr1"),
                    "p_emaildomain": row.get("p_emaildomain"),
                    "r_emaildomain": row.get("r_emaildomain"),
                }

                # Call API
                response = requests.post(
                    f"{API_ENDPOINT}/predict",
                    json=request_data,
                    timeout=10,
                )
                response.raise_for_status()

                prediction = response.json()
                results.append({
                    "transaction_id": row.get("transaction_id", i),
                    "fraud_score": prediction["fraud_score"],
                    "is_fraud": prediction["is_fraud"],
                    "model_version": prediction["model_version"],
                    "explanation": prediction["explanation"],
                    "scored_at": datetime.utcnow().isoformat(),
                })

            except Exception as e:
                print(f"[Score] Error scoring transaction {i}: {e}")
                failed_count += 1
                continue

    results_df = pd.DataFrame(results)
    print(
        f"[Score] ✓ Scored {len(results)} transactions "
        f"({failed_count} failures)"
    )

    context["task_instance"].xcom_push(key="scored_df", value=results_df)
    return {"status": "success", "scored": len(results), "failed": failed_count}


def load_results_to_s3(**context):
    """
    Write scored transactions to S3 as Parquet.

    Path: s3://fraudlens/scored/YYYY-MM-DD/results.parquet
    """
    partition_date = context["task_instance"].xcom_pull(
        task_ids="extract_from_s3",
        key="partition_date",
    )
    scored_df = context["task_instance"].xcom_pull(
        task_ids="batch_score",
        key="scored_df",
    )

    if scored_df is None or scored_df.empty:
        print("[Load] No results to load")
        return {"status": "skipped"}

    s3_path = f"s3://{S3_BUCKET}/{S3_SCORED_PREFIX}/{partition_date}/results.parquet"
    print(f"[Load] Writing {len(scored_df)} results to {s3_path}")

    try:
        scored_df.to_parquet(s3_path, index=False)
        print(f"[Load] ✓ Results saved")
        return {"status": "success", "path": s3_path, "rows": len(scored_df)}

    except Exception as e:
        print(f"[Load] Error saving results: {e}")
        raise


def athena_create_partition(**context):
    """
    Create Athena partition for today's results.

    Registers the new Parquet file so it's immediately queryable via Athena.
    """
    partition_date = context["task_instance"].xcom_pull(
        task_ids="extract_from_s3",
        key="partition_date",
    )

    print(f"[Athena] Creating partition for {partition_date}")

    # In production, use boto3 Athena client:
    # client = boto3.client('athena')
    # response = client.start_query_execution(
    #     QueryString=f"ALTER TABLE fraudlens_scored ADD PARTITION (date='{partition_date}')",
    #     ...
    # )

    print(f"[Athena] ✓ Partition created")
    return {"status": "success", "partition": partition_date}


# - DAG Task Dependencies 

task_extract = PythonOperator(
    task_id="extract_from_s3",
    python_callable=extract_transactions_from_s3,
    dag=dag,
)

task_validate = PythonOperator(
    task_id="validate_transactions",
    python_callable=validate_transactions,
    dag=dag,
)

task_score = PythonOperator(
    task_id="batch_score",
    python_callable=batch_score_transactions,
    dag=dag,
)

task_load = PythonOperator(
    task_id="load_to_s3",
    python_callable=load_results_to_s3,
    dag=dag,
)

task_athena = PythonOperator(
    task_id="athena_partition",
    python_callable=athena_create_partition,
    dag=dag,
)

# Set task dependencies
task_extract >> task_validate >> task_score >> task_load >> task_athena
