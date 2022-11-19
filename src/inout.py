
import numpy as np
import pickle
import json

def save_trained_model(path: str, model):
    """Saves a trained model as pickle file.
    
    Parameters
    --------------
    path: str
        The path where we want to save the model.
        Needs to come with the extension.
    model:
        The model to save
    """
    with open(path, 'wb') as file:
        pickle.dump(model, file)

def load_model_from_pickle(path: str):
    """Loads a pretrained model. 
    
    Parameters
    ---------
    path: str
        Path (with extension) to the file where the model is saved
    
    Returns
    ----------
    The loaded model
    """
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_predictions(filename: str, predictions: np.ndarray):
    """
    Saves a 1D numpy array both as csv and json files.
    
    Parameters
    ----------
    filename: str
        The name that both files will have, without extension.
    predictions: np.ndarray
        A 1D array containing the labels of the predictions.
    """
    if isinstance(predictions, np.ndarray):
        save_as_csv(f'{filename}.csv', predictions)
        save_as_json(f'{filename}.json', predictions)
    else:
        raise TypeError('predictions must be of class numpy.ndarray')

def save_as_csv(path: str, predictions: np.ndarray, header: str='target'):
    """
    Saves a 1D numpy array as csv file.
    
    Parameters
    ----------
    filename: str
        The path where the csv will be saved, with extension.
    predictions: np.ndarray
        A 1D array containing the labels of the predictions.
    header: str
        The header wanted on the csv file. Default: 'target'
    """
    np.savetxt(path, predictions, delimiter=';', fmt='%i', header=header)

def save_as_json(path: str, predictions: np.ndarray):
    """
    Saves a 1D numpy array as json file.
    
    The json file will start by {'target':}, containing there a 
    dictionary where the keys are the indices and the values are
    the actual predictions for each index
    Parameters
    ----------
    filename: str
        The path where the csv will be saved, with extension.
    predictions: np.ndarray
        A 1D array containing the labels of the predictions.
    header: str
        The header wanted on the csv file. Default: 'target'
    """
    pred_dict = {'target':{}}
    pred_dict['target'] = dict(enumerate(predictions))
    with open(path, 'w') as file:
        json.dump(pred_dict, file, cls=NpEncoder)  

class NpEncoder(json.JSONEncoder):
    """
    Class to encode the values of a NumPy array as integers
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)

