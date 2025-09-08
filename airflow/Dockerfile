
FROM apache/airflow:2.9.2-python3.11
USER airflow
COPY --chown=airflow:0 requirements.txt /requirements.txt
RUN pip install --no-cache-dir --upgrade-strategy only-if-needed -r /requirements.txt
ENV PATH="/home/airflow/.local/bin:${PATH}"
