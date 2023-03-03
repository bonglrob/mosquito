"""
CSE 163 Final Project
Kyoko Kurihara, Name here?

This file implements functions for mosquito prediction. ...
"""

import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt


def get_path(filename: str) -> str:
    """
    This function takes a file name and returns the file path.
    """
    return os.path.join('./dataset', filename)


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


def generate_city_df() -> pd.DataFrame:
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


def get_df_pop(file_name):
    data = pd.DataFrame(pd.read_excel(file_name))
    return data


def clean_up_pop_df():
    # 1947-1970 data
    pop4769 = get_df_pop(get_path('popca_4769.xlsx'))
    pop4769 = pop4769.loc[2:59, :]
    years = [str(i) for i in range(1947, 1971)]
    columns = ["County", "1940"] + years
    columns.insert(5, "1950_ap")
    columns.insert(16, "1960_ap")
    pop4769.columns = columns
    pop4769 = pop4769.drop(columns=['1950_ap', '1960_ap', '1970'])

    pop7080 = get_df_pop(get_path('popca_7080.xlsx'))
    pop7080 = pop7080.loc[:, ['Unnamed: 0', 'Unnamed: 1', 'Unnamed: 2']]
    pop7080.columns = ['County', 'Year', 'Population']
    print(pop7080)
    pop_all = pop4769.copy()
    pop_1970 = []
    for i in range(58):
        pop_id = 15 + i * 13
        pop_1970.append(pop7080.loc[15, 2])
    print(pop_all)
    # append a list of 1970 based on county
    # do the same for all
    pass



def main() -> None:
    # read files
    # mosquito1 = get_df_m(get_path('Aedes_aegypti_occurrence.csv'))
    # mosquito2 = get_df_m(get_path('Anopheles_quadrimaculatus_occurrence.csv'))
    # mosquito3 = get_df_m(get_path('Culex_tarsalis_occurrence.csv'))
    # print(mosquito3)

    # city_data = generate_city_df()
    clean_up_pop_df()

    

    # question 1

    # question 2

    # question 3


if __name__ == '__main__':
    main()
