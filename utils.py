import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
import os

def export_numpy_array(array: np.ndarray, filename: str) -> None:
    """
    Exports a NumPy array to a file.

    Parameters:
    array (np.ndarray): The NumPy array to be exported.
    filename (str): The file path where the array will be saved.

    Returns:
    None
    """
    np.save(filename, array)

def import_numpy_array(filename: str) -> np.ndarray:
    """
    Imports a NumPy array from a file.

    Parameters:
    filename (str): The file path where the array is saved.

    Returns:
    np.ndarray: The imported NumPy array.
    """
    return np.load(filename)

def pickle_export(data, path = "population data/", name = ""):
    os.makedirs(os.path.dirname(path+name), exist_ok=True)
    with open(path + name, "wb") as f:
        pickle.dump(data, f)
        
def pickle_import(path = "population data/", name = ""):
    with open(path + name, "rb") as f:
        data = pickle.load(f)
    return data
