
#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p airflow/dags airflow/logs models
UID_HOST="$(id -u)"
if grep -q '^AIRFLOW_UID=' .env; then
  sed -i.bak "s/^AIRFLOW_UID=.*/AIRFLOW_UID=${UID_HOST}/" .env
else
  echo "AIRFLOW_UID=${UID_HOST}" >> .env
fi

(sudo chown -R "${UID_HOST}:0" airflow models || true)
(sudo chmod -R g+rwX airflow models || true)
echo "Permissions fixed. AIRFLOW_UID=${UID_HOST}"
