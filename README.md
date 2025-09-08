# Taller 3 – Orquestación con Airflow y Despliegue de API para Inferencia

## Integrantes
- **Yibby González**  
- **Sebastián Ruiz**  
- **Adrián Téllez**

---

## 🎯 Objetivo
El objetivo de este taller es implementar un **pipeline completo de Machine Learning** utilizando **Apache Airflow**, desde la ingesta de datos hasta el despliegue de un modelo entrenado mediante un servicio de inferencia expuesto con **FastAPI**.  

---

## 📦 Arquitectura de la solución
El proyecto levanta múltiples servicios en **Docker Compose**:

- **Postgres** → Base de datos de metadatos de Airflow.  
- **MariaDB** → Base de datos para almacenar los datos de *penguins*.  
- **Redis** → Broker para el scheduler de Airflow.  
- **Airflow Webserver** → Interfaz web de Airflow (DAGs y monitoreo).  
- **Airflow Scheduler** → Encargado de planificar y ejecutar las tareas.  
- **Airflow Worker** → Ejecutor de las tareas del DAG.  
- **API FastAPI** → Servicio que carga el modelo entrenado y permite realizar inferencia.  

### 🖼️ Diagrama de Arquitectura
```mermaid
flowchart LR
    subgraph Airflow[Airflow Cluster]
        A[Airflow Webserver] --> B[Scheduler]
        B --> C[Worker]
        C -->|Ejecuta DAGs| D[MariaDB: penguins data]
        B -->|Metadatos| E[Postgres]
        B -->|Mensajería| F[Redis]
    end

    C --> G[Modelo entrenado /models/model.pkl]
    G --> H[FastAPI Service]

    H -->|Predicciones| User[Cliente / cURL / Swagger UI]
```

---

## ⚙️ Evidencia de ejecución

Al levantar el proyecto con `docker compose up --build`, se puede verificar con `docker ps` los servicios corriendo:

```
NAME                                         IMAGE                                      COMMAND                  SERVICE             CREATED         STATUS                   PORTS
penguins-airflow-final-airflow-scheduler-1   penguins-airflow-final-airflow-scheduler   "/usr/bin/dumb-init …"   airflow-scheduler   9 minutes ago   Up 5 minutes             8080/tcp
penguins-airflow-final-airflow-webserver-1   penguins-airflow-final-airflow-webserver   "/usr/bin/dumb-init …"   airflow-webserver   9 minutes ago   Up 3 minutes             0.0.0.0:8081->8080/tcp, [::]:8081->8080/tcp
penguins-airflow-final-airflow-worker-1      penguins-airflow-final-airflow-worker      "/usr/bin/dumb-init …"   airflow-worker      9 minutes ago   Up 3 minutes             8080/tcp
penguins-airflow-final-api-1                 penguins-airflow-final-api                 "uvicorn main:app --…"   api                 9 minutes ago   Up 3 minutes             0.0.0.0:8080->8000/tcp, [::]:8080->8000/tcp
penguins-airflow-final-penguins-db-1         mariadb:11                                 "docker-entrypoint.s…"   penguins-db         9 minutes ago   Up 9 minutes (healthy)   0.0.0.0:3306->3306/tcp, [::]:3306->3306/tcp
penguins-airflow-final-postgres-1            postgres:15-alpine                         "docker-entrypoint.s…"   postgres            9 minutes ago   Up 9 minutes (healthy)   0.0.0.0:5432->5432/tcp, [::]:5432->5432/tcp
penguins-airflow-final-redis-1               redis:7-alpine                             "docker-entrypoint.s…"   redis               9 minutes ago   Up 9 minutes             6379/tcp
```

📸 **Evidencia – Airflow corriendo con el DAG**  
![Airflow UI](f8c15c5f-420d-4924-bd4c-13cf45d634a7.png)  

📸 **Evidencia – FastAPI con predicciones**  
![FastAPI UI](9dcbc2ad-4e81-43ca-85fa-86f57b39d5ca.png)  

---

## 🚀 Ejecución del pipeline

1. **Levantar el entorno**  
   ```bash
   docker compose up --build
   ```

2. **Acceder a Airflow**  
   - URL: [http://localhost:8081](http://localhost:8081)  
   - Usuario: `admin`  
   - Contraseña: `admin`

3. **Ejecutar el DAG `penguins_etl_train`**  
   Este DAG realiza las siguientes tareas:  
   - Borrar contenido de la base de datos.  
   - Cargar los datos de *penguins* sin preprocesar.  
   - Realizar el preprocesamiento.  
   - Entrenar un modelo de clasificación y guardarlo en `/models/model.pkl`.  

---

## 🧪 Consumo de la API de inferencia

Una vez entrenado el modelo, el servicio de API en **FastAPI** queda disponible en:  
👉 [http://localhost:8080](http://localhost:8080)

### 1. Probar que el servicio está activo
```bash
curl http://localhost:8080/
```

### 2. Realizar una predicción
```bash
curl -X POST http://localhost:8080/predict   -H 'Content-Type: application/json'   -d '{
        "island": "Torgersen",
        "bill_length_mm": 39.1,
        "bill_depth_mm": 18.7,
        "flipper_length_mm": 181,
        "body_mass_g": 3750,
        "sex": "Male"
      }'
```

### 3. Respuesta esperada
```json
{
  "prediction": "Adelie",
  "probabilities": {
    "Adelie": 0.86,
    "Gentoo": 0.14,
    "Chinstrap": 0.01
  }
}
```

---

## ✅ Conclusiones
- Se logró implementar un **pipeline de ML orquestado con Airflow**.  
- Los datos fueron procesados y almacenados en MariaDB.  
- Se entrenó un modelo que luego se expone como **servicio de inferencia con FastAPI**.  
- Todo el ecosistema corre de manera integrada en **Docker Compose**.  
