from kdd_pred.logger import logging
from kdd_pred.exception import KDDEXCEPTION
from typing import Optional
import os, sys
import pandas as pd
import numpy as np
from kdd_pred.predictor import ModelResolver
from kdd_pred import utils


PREDICTION_DIR = "Prediction"


def strat_batch_prediction(input_file_path):
    try:
        
        os.makedirs(PREDICTION_DIR, exist_ok=True)
        model_resolver=ModelResolver(model_registry="saved_models")

        # load data:
        data = pd.read_csv(input_file_path)
        data = data.copy()
        data = data.dropna(axis=0)

        data['class'] = data['class'].apply(lambda x: 1 if x == 'anomaly' else 0)

        transformer = utils.load_object(file_path=model_resolver.get_latest_save_transform_path())
        
        #input_feature_names = list(transformer.get_feature_names_out())
        #input_arr = transformer.transform(data[input_feature_names])

        input_arr = transformer.transform(data)
    

        model = utils.load_object(file_path=model_resolver.get_latest_save_model_path())
        prediction = model.predict(input_arr)

        data['prediction'] = np.round(prediction).astype(int)

        prediction_file_name = os.path.join(PREDICTION_DIR, "prediction_file.csv")
        data.to_csv(path_or_buf=prediction_file_name, index=False, header=True)

        return prediction_file_name

    except Exception as e:
        raise KDDEXCEPTION(e, sys) from e