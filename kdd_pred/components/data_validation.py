from kdd_pred.logger import logging
from kdd_pred.exception import KDDEXCEPTION
from kdd_pred.entity import config_entity, artifacts_entity
import pandas as pd
import numpy as np
import sys
from typing import Optional


class DataValidation:

    def __init__(self, data_validation_config:config_entity.DataValidationConfig,
                    data_ingestion_artifacts:artifacts_entity.DataIngestionArtifact):
        try:
            
            self.data_validation_config=data_validation_config
            self.data_ingestion_artifacts=data_ingestion_artifacts

        except Exception as e:
            raise KDDEXCEPTION(e,sys)



    def drop_null_value_rows(self, df:pd.DataFrame)->Optional[pd.DataFrame]:
        try:
            
            df = df.dropna(axis=0)
            
            logging.info(f"now shape of df : {df.shape}")

            return df

        except Exception as e:
            raise KDDEXCEPTION(e,sys)

    def drop_index_column(self, df:pd.DataFrame)->Optional[pd.DataFrame]:
        try:

            if 'index' in df.columns:

                df = df.drop(columns=['index'], axis=1)
            
            logging.info(f"now shape of df : {df.shape}")

            return df

        except Exception as e:
            raise KDDEXCEPTION(e,sys)
        


    def convert_cols_float_to_int(self, df:pd.DataFrame)->Optional[pd.DataFrame]:
        try:

            for cols in df.columns:
                if df[cols].dtype == float:
                    df[cols] = df[cols].astype(int)
            
            logging.info(f"Data types converted from float to int")

            return df

        except Exception as e:
            raise KDDEXCEPTION(e,sys)
               

    def initiate_data_validation(self)->artifacts_entity.DataValidationArtifact:
        try:
            logging.info("data validation started........")

            train_df = pd.read_csv(self.data_ingestion_artifacts.train_data_path)
            test_df = pd.read_csv(self.data_ingestion_artifacts.test_data_path)

            logging.info(f"dropping null values")
            train_df = self.drop_null_value_rows(df=train_df)
            test_df = self.drop_null_value_rows(df=test_df)
        
            # drop_index_column
            logging.info(f"drop_index_column ")
            train_df=self.drop_index_column(df=train_df)
            test_df=self.drop_index_column(df=test_df)

            # Convert columns if they are in float into int
            train_df = self.convert_cols_float_to_int(df=train_df)
            test_df = self.convert_cols_float_to_int(df=test_df)

            # SAve the dataset into artifacts
            train_df.to_csv(path_or_buf=self.data_validation_config.valid_train_path, index=False, header=True)
            test_df.to_csv(path_or_buf=self.data_validation_config.valid_test_path, index=False, header=True)


            logging.info("data validation is almost done")
            

            data_validation_artifact = artifacts_entity.DataValidationArtifact(
                valid_train_path = self.data_validation_config.valid_train_path,
                valid_test_path= self.data_validation_config.valid_test_path)
            
            logging.info("returning data_validation_artifact")


            return data_validation_artifact

        except Exception as e:
            raise KDDEXCEPTION(e,sys)

