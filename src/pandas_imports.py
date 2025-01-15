import pandas as pd
import numpy as np


def get_string_entries(df):
    """
    df: pandas dataframe object
    Returns: list of values which contains str variables, makes sure all other values are floats or integers
    """
    keys_string = []
    mapping_list = {} #a dictionary which holds for the keys of keys_string a list which can be used to transform str values to numerical values
    for k in df.keys():
        if type(df[k][0]) == str:
            keys_string.append(k)
            mapping_list[k] =  list(set(df[k].tolist()))
        elif type(df[k][0]) == np.int64 or type(df[k][0]) == np.float64:
            pass
        else:
            raise ValueError('Dataframe pf contains entry which is not of type str, int, or float')
    return keys_string,mapping_list

def transform_to_numerical(df):
    """
    transform all string values in pf to numerical values
    df: pandas dataframe object
    Returns: pandas dataframe
    """
    string_keys,mapping_list = get_string_entries(df)
    for k in string_keys:
        df[k] = df[k].apply(lambda x: mapping_list[k].index(x))

    return df