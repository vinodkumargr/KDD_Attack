from kdd_pred.exception import KDDEXCEPTION
from kdd_pred.logger import logging
from kdd_pred.components.data_ingestion import DataIngestion
from kdd_pred.components.data_validation import DataValidation
from kdd_pred.components.data_transformation import DataTransformation
from kdd_pred import utils
from kdd_pred.entity import config_entity, artifacts_entity
from kdd_pred.predictor import ModelResolver
import os, sys, re
import pandas as pd
import numpy as np




class ModelPusher:

    def __init__(self, model_pusher_config:config_entity.ModelPusherConfig,
                 data_transformation_artifact:artifacts_entity.DataTransformationArtifact,
                 model_trainer_artifact:artifacts_entity.ModelTrainerArtifact):
        
        try:

            self.model_pusher_config = model_pusher_config
            self.data_transformation_artifacts = data_transformation_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self.model_resolver = ModelResolver(model_registry=self.model_pusher_config.saved_model_dir)

        except Exception as e:
            raise KDDEXCEPTION(e, sys)
        

    def initiate_model_pusher(self)-> artifacts_entity.ModelPusherArtifact:
        
        try:
            logging.info("model pusher is started.....")

            logging.info("reading(unpickling) transformer and model")
            transformer = utils.load_object(file_path=self.data_transformation_artifacts.pre_process_object_path)
            data = utils.load_numpy_array_data(file_path=self.data_transformation_artifacts.transform_train_path)
            model = utils.load_object(file_path=self.model_trainer_artifact.model_path)

            # model pusher dir
            utils.save_object(file_path=self.model_pusher_config.pusher_transform_path, obj=transformer)
            utils.save_object(file_path=self.model_pusher_config.pusher_model_path, obj=model)

            # save model in dir:
            logging.info("saving transformer and model file into saved folder...")
            transform_path = self.model_resolver.get_latest_save_transform_path()
            model_path = self.model_resolver.get_latest_save_model_path()
            data_path = self.model_resolver.get_latest_save_data_path()
            
            utils.save_object(file_path=transform_path, obj=transformer)
            utils.save_object(file_path=model_path, obj=model)
            utils.save_object(file_path=data_path, obj=data)

            model_pusher_artifact = artifacts_entity.ModelPusherArtifact(
                                                        pusher_model_dir=self.model_pusher_config.pusher_model_dir,
                                                        saved_model_dir=self.model_pusher_config.saved_model_dir)

            return model_pusher_artifact


        except Exception as e:
            raise KDDEXCEPTION(e, sys)
