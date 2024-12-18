from airflow.models import DAG

from airflow.operators.python import PythonOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator

from datetime import datetime

from utilization.scraping_link import ScrapingLink
from utilization.scraping_data import ScrapingData
from utilization.cleaning_data import CleaningData
from utilization.feature_engineering import FeatureEngineering
# from utilization.modeling import Modeling

default_args= {
    'owner': 'aziz',
    'start_date': datetime(2024, 12, 16),
}

with DAG(
    "house_prediction_without_modeling",
    description='End-to-end ML Pipeline House Prediction',
    schedule_interval='@monthly',
    default_args=default_args, 
    catchup=False) as dag:

    # task: 1
    scraping_link = PythonOperator(
        task_id='ScrapingLink',
        python_callable=ScrapingLink
    )    

    # task: 2
    scraping_data = PythonOperator(
        task_id='ScrapingData',
        python_callable=ScrapingData
    )
    
    # task: 3
    cleaning_data = PythonOperator(
        task_id='CleaningData',
        python_callable=CleaningData
    )
        
    # task: 4
    feature_engineering = PythonOperator(
        task_id='FeatureEngineering',
        python_callable=FeatureEngineering
    )

    # # task: 5
    # modeling = PythonOperator(
    #     task_id='Modeling',
    #     python_callable=Modeling
    # )
    scraping_link >> scraping_data >> cleaning_data >> feature_engineering #>> modeling