"""
CSE 163 Final Project
Kyoko Kurihara, Name here?

This file implements functions for mosquito prediction. ...
"""

import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt


def get_df_m(file_path: str) -> pd.DataFrame:
    """
    This function takes a file name (str) of one of mosquito occurence
    datasets and returns the dataframe.
    """
    # is there any other way "low_memory=False"
    data = pd.read_csv(file_path, delimiter='\t', low_memory=False)
    # select columns
    data = data.loc[data['countryCode'] == 'US',
                    ['gbifID', 'species', 'countryCode', 'locality',
                     'stateProvince', 'occurrenceStatus', 'individualCount',
                     'decimalLatitude', 'decimalLongitude', 'elevation',
                     'elevationAccuracy', 'depth', 'depthAccuracy', 'day',
                     'month', 'year']
                    ]
    return data


def get_df_t(file_path: str) -> pd.DataFrame:
    """
    This function takes a file name (str) of one of temperature datasets
    and returns the dataframe.
    """
    data = pd.read_csv(file_path)
    data.columns = ["Date", "Temp", "Anomaly"]
    data = data.loc[4:, "Date":"Temp"]
    return data


def get_df_p(file_path: str) -> pd.DataFrame:
    """
    This function takes a file name (str) of one of precipitation datasets
    and returns the file in dataframe format.
    """
    data = pd.read_csv(file_path)
    data.columns = ["Date", "Prec", "Anomaly"]
    data = data.loc[4:, "Date":"Prec"]
    return data


def city_df() -> pd.DataFrame:
    """
    This function returns a dataframe where all datasets on city tempereature
    and precipitation are combined.
    """
    cities = ["Eureka", "Fresno", "Los_Angeles", "Sacramento",
              "San_Diego", "San_Francisco"]
    result = None
    for city in cities:
        # read files
        temp_path = './dataset/' + city + '_temp.csv'
        prec_path = './dataset/' + city + '_prec.csv'
        temp_df = get_df_t(temp_path)
        prec_df = get_df_p(prec_path)

        # merge files
        if result is None:
            result = temp_df.merge(prec_df, left_on="Date",
                                   right_on="Date", how="outer")
            result.rename(columns={"Temp": "Temp_" + city,
                                   "Prec": "Prec_" + city}, inplace=True)

        else:
            result = result.merge(temp_df, left_on="Date",
                                  right_on="Date", how="outer")
            result = result.merge(prec_df, left_on="Date",
                                  right_on="Date", how="outer")
            result.rename(columns={"Temp": "Temp_" + city,
                                   "Prec": "Prec_" + city}, inplace=True)
    return result


def main() -> None:
    # read files
    mosquito1 = get_df_m(os.path.join('./dataset',
                         'Aedes_aegypti_occurrence.csv'))
    mosquito2 = get_df_m(os.path.join('./dataset',
                         'Anopheles_quadrimaculatus_occurrence.csv'))
    mosquito3 = get_df_m(os.path.join('./dataset',
                         'Culex_tarsalis_occurrence.csv'))
    print(mosquito3)

    city_data = city_df()

    # question 1

    # question 2

    # question 3


if __name__ == '__main__':
    main()
