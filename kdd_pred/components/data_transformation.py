from kdd_pred.exception import KDDEXCEPTION
from kdd_pred.logger import logging
from kdd_pred import entity
from kdd_pred.entity import config_entity, artifacts_entity
from kdd_pred import utils
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

class DataTransformation:

    def __init__(self, data_transformation_config:config_entity.DataTransformationConfig,
                        data_validation_artifacts:artifacts_entity.DataValidationArtifact):
        try:
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifacts = data_validation_artifacts
        except Exception as e:
            raise KDDEXCEPTION(e, sys)

    def get_transform(self):
        try:
            # Define categorical columns
            categorical_columns = ['protocol_type', 'service', 'flag']

            # Define all columns
            all_columns = entity.input_columns

            numerical_columns = [col for col in all_columns if col not in categorical_columns]



            one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
            scaler = StandardScaler()
            pca = PCA(n_components=11)

            # Create a column transformer for numerical and categorical columns
            preprocessor = ColumnTransformer([
                ('categorical', one_hot_encoder, categorical_columns),
                ('numerical', scaler, numerical_columns)
            ])

            # Create a pipeline with preprocessing steps and PCA
            transformer = Pipeline([
                ('preprocessor', preprocessor),
                ('pca', pca)
            ])

            return transformer

        except Exception as e:
            raise KDDEXCEPTION(e, sys)

    def initiate_data_transformation(self) -> artifacts_entity.DataTransformationArtifact:
        try:
            logging.info("Starting data transformation...")

            # Read train and test data
            train_df = pd.read_csv(self.data_validation_artifacts.valid_train_path)
            test_df = pd.read_csv(self.data_validation_artifacts.valid_test_path)

            train_df.dropna(axis=0, inplace=True)
            test_df.dropna(axis=0, inplace=True)

            # Get input and output features
            input_train_features, out_train_feature = train_df.drop(columns=['class'], axis=1), train_df['class']
            input_test_features, out_test_feature = test_df.drop(columns=['class'], axis=1), test_df['class']


            # Get transformer object
            transformer = self.get_transform()

            # Perform transformation on train and test data
            input_train_preprocessing_arr = transformer.fit_transform(input_train_features)
            input_test_preprocessing_arr = transformer.transform(input_test_features)

            # pca = PCA(n_components=11)
            # input_train_preprocessing_arr = pca.fit_transform(input_train_preprocessing_arr)
            # input_test_preprocessing_arr = pca.transform(input_test_preprocessing_arr)

            logging.info(f"{input_train_preprocessing_arr}")            

            # Combine input array and output feature
            train_arr = np.c_[input_train_preprocessing_arr, np.array(out_train_feature)]
            test_arr = np.c_[input_test_preprocessing_arr, np.array(out_test_feature)]

            # Save transformed data
            utils.save_numpy_array_data(file_path=self.data_transformation_config.transform_train_path, array=train_arr)
            utils.save_numpy_array_data(file_path=self.data_transformation_config.transform_test_path, array=test_arr)
        
            # Save preprocessing object
            utils.save_object(file_path=self.data_transformation_config.pre_process_object_path, obj=transformer)

            # Save data from data validation
            data_obj = pd.read_csv(self.data_validation_artifacts.valid_train_path)
            utils.save_object(file_path=self.data_transformation_config.single_pred_data_path, obj=data_obj)

            data_transformation_artifact = artifacts_entity.DataTransformationArtifact(
                transform_train_path=self.data_transformation_config.transform_train_path,
                transform_test_path=self.data_transformation_config.transform_test_path,
                pre_process_object_path=self.data_transformation_config.pre_process_object_path
            )

            logging.info("Data transformation completed.")
            return data_transformation_artifact

        except Exception as e:
            raise KDDEXCEPTION(e, sys)
