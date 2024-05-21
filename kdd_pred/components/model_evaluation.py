from kdd_pred.exception import KDDEXCEPTION
from kdd_pred.logger import logging
from kdd_pred import utils
from kdd_pred.entity import config_entity, artifacts_entity
from kdd_pred.predictor import ModelResolver
import os, sys, re
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score



class ModelEvaluation:
    def __init__(self, model_evaluation_config:config_entity.ModeEvaluationConfig,
                 data_ingestion_artifacts:artifacts_entity.DataIngestionArtifact,
                 data_validation_artifacts:artifacts_entity.DataValidationArtifact,
                 data_transformation_artifacts:artifacts_entity.DataTransformationArtifact,
                 model_trainer_artifacts:artifacts_entity.ModelTrainerArtifact):
        
        try:

            self.model_evaluation_config=model_evaluation_config
            self.data_ingestion_artifact = data_ingestion_artifacts
            self.data_validation_artifacts = data_validation_artifacts
            self.data_transformation_artifacts = data_transformation_artifacts
            self.model_trainer_artifacts = model_trainer_artifacts
            self.model_resolver = ModelResolver()

        except Exception as e:
            raise KDDEXCEPTION(e, sys)
        

    def initiate_model_evaluation(self)-> artifacts_entity.ModelEvaluationArtifact:
        try:

            logging.info("Model evaluation started ......")
            latest_dir_path = self.model_resolver.get_latest_dir_path()

            if latest_dir_path == None:  # if the model accuracy is not increased then it will not creates new model dirs 
                model_evaluation_artifact=artifacts_entity.ModelEvaluationArtifact(model_eccepted=True, improved_accuracy=None)

                logging.info(f"model_evaluation_artifact : {model_evaluation_artifact}")

                return model_evaluation_artifact
        

            #find previous/old model location
            logging.info("finding old model path...")
            old_transformer_path = self.model_resolver.get_latest_save_transform_path()
            old_model_path = self.model_resolver.get_latest_model_path()
            
            # read previous model
            logging.info("reading old model...")
            test_data = utils.load_numpy_array_data(self.data_transformation_artifacts.transform_test_path)
            old_transformer = utils.load_object(old_transformer_path)
            old_model = utils.load_object(file_path=old_model_path)

            # read current/new model
            logging.info("reading new model...")
            current_transformer = utils.load_object(self.data_transformation_artifacts.pre_process_object_path)
            current_model = utils.load_object(file_path=self.model_trainer_artifacts.model_path)

            # reading old test data and predicting for old model
            old_test_data = test_data
            old_x_test, old_y_test = old_test_data[:, :-1] , old_test_data[:, -1]

            old_model_y_pred = old_model.predict(old_x_test)

            #previous model accuracy_score
            logging.info("comapring models....")
            prevoius_model_accuracy_score = accuracy_score(y_true=old_y_test, y_pred=old_model_y_pred)
            logging.info(f"previous model accuracy_score : {prevoius_model_accuracy_score}")


            # reading new test data and predicting for new model
            new_x_test_data = test_data
            new_x_test, new_y_test = new_x_test_data[:, :-1] , new_x_test_data[:, -1]

            new_model_y_pred = current_model.predict(new_x_test)

            current_model_accuracy_score = accuracy_score(y_true=new_y_test, y_pred=new_model_y_pred)
            logging.info(f"current_model accuracy_score : {current_model_accuracy_score}")

               

            model_evaluation_artifact = artifacts_entity.ModelEvaluationArtifact(
                                                    model_eccepted=True, 
                                                    improved_accuracy=current_model_accuracy_score - prevoius_model_accuracy_score)

            return model_evaluation_artifact


        except Exception as e:
            raise KDDEXCEPTION(e, sys)