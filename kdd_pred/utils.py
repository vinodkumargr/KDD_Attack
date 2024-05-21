import pandas as pd
import numpy as np
import os, sys
from kdd_pred.exception import KDDEXCEPTION
from kdd_pred.logger import logging
import pickle



def save_numpy_array_data(file_path: str, array: np.array):

    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise KDDEXCEPTION(e, sys)
    

def load_numpy_array_data(file_path: str):
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj, allow_pickle=True)
    except Exception as e:
        raise KDDEXCEPTION(e, sys)
    


def save_object(file_path: str, obj: object) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise KDDEXCEPTION(e, sys)
    

def load_object(file_path:str)->object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"file path : {file_path} not exists...")
        with open(file_path , "rb") as file_obj:
            return pickle.load(file_obj)
        
    except Exception as e:
        raise KDDEXCEPTION(e, sys)
    
