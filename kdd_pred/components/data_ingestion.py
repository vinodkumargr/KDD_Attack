import pandas as pd
import numpy as np
import shutil
import sys
from kdd_pred import utils
from kdd_pred.exception import KDDEXCEPTION
from kdd_pred.logger import logging
from kdd_pred import entity
from kdd_pred.entity import config_entity, artifacts_entity
from sklearn.model_selection import train_test_split


class DataIngestion:
    
    def __init__(self, data_ingestion_config: config_entity.DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise KDDEXCEPTION(e, sys)
        

    def start_data_ingestion(self)-> artifacts_entity.DataIngestionArtifact:
        try:

            logging.info("Starting data ingestion")

            df = pd.read_csv("/home/vinod/projects_1/KDD_END_2_END/data/KDDTrain_.csv")


            df['class'] = df['class'].apply(lambda x: 1 if x == 'anomaly' else 0)

            x_train, x_test = train_test_split(df, test_size=0.2)

            logging.info("storing train and test data")
            x_train.to_csv(path_or_buf = self.data_ingestion_config.train_path, index=False, header=True)
            x_test.to_csv(path_or_buf = self.data_ingestion_config.test_path, index=False, header=True)           

            
            #preparing artifacts folder:
            data_ingestion_artifact = artifacts_entity.DataIngestionArtifact(
                train_data_path=self.data_ingestion_config.train_path,
                test_data_path=self.data_ingestion_config.test_path)

            return data_ingestion_artifact

        except Exception as e:
            raise KDDEXCEPTION(e,sys)



