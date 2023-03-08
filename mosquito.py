"""
CSE 163 Final Project
Kyoko Kurihara, Robert Bonglamphone, Name here?

This file implements functions for mosquito prediction. ...
"""

import os
import pandas as pd
from shapely.geometry import Point
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from typing import List
from shapely.geometry import Point
import plotly.express as px


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
                    ['species', 'countryCode', 'locality',
                     'stateProvince', 'individualCount',
                     'decimalLatitude', 'decimalLongitude', 'elevation',
                     'elevationAccuracy', 'depth', 'depthAccuracy', 'day',
                     'month', 'year']
                    ]
    data["individualCount"] = data["individualCount"].fillna(1)
    return data


def ca_geomosquito(df: pd.DataFrame):
    """
    This function takes DataFrame and returns filtered GeoDataFrame.
    """
    result = get_geometry(filter_ca(df))
    return result


def filter_ca(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function takes a mosquito occurence
    dataframes and returns the dataframe only with rows including CA.
    """
    # filter rows
    mask = df['stateProvince'] == 'California'
    return df[mask]



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


def pop50():
    # 1947-1970 data
    pop4769 = get_df_pop(get_path('popca_4769.xlsx'))
    pop4769 = pop4769.loc[2:59, :]
    years = [str(i) for i in range(1947, 1971)]
    columns = ["County", "1940"] + years
    columns.insert(5, "1950_ap")
    columns.insert(16, "1960_ap")
    pop4769.columns = columns
    pop4769 = pop4769.drop(columns=['1950_ap', '1960_ap', '1970'])
    return pop4769


def combine_pop_df():
    pop70 = clean_up_pop_df('popca_7080.xlsx', 15, 9)
    pop80 = clean_up_pop_df('popca_8090.xlsx', 15, 9)
    pop90 = clean_up_pop_df('popca_9000.xlsx', 15, 9)
    pop00 = clean_up_pop_df('popca_0010.xlsx', 20, 10)
    pop00 = pop00.drop(pop00[pop00['Year'] == 1999].index)
    pop10 = clean_up_pop_df('popca_1021.xlsx', 20, 12)
    pop10 = pop10.loc[pop10['Year'] != 'Census 2010']
    pop10['Year'] = pop10['Year'].replace('Apr-Jun 2010', 2010)

    pop_50 = pop50()
    pop_70 = change_style(pop70)
    pop_80 = change_style(pop80)
    pop_90 = change_style(pop90)
    pop_00 = change_style(pop00)
    pop_10 = change_style(pop10)

    # merge
    pop_all = pop_50.merge(pop_70, left_on='County', right_on='County', how='left')
    pop_all = pop_all.merge(pop_80, left_on='County', right_on='County', how='left')
    pop_all = pop_all.merge(pop_90, left_on='County', right_on='County', how='left')
    pop_all = pop_all.merge(pop_00, left_on='County', right_on='County', how='left')
    pop_all = pop_all.merge(pop_10, left_on='County', right_on='County', how='left')

    # print(pop_all)
    return pop_all


def change_style(pop: pd.DataFrame) -> pd.DataFrame:
    result = {}
    result['County'] = []

    for county in list(set(pop['County'])):
        result['County'].append(county)
        pop_copy = pop[pop['County'] == county]
        for year in list(pop_copy['Year']):
            year = int(year)
            row_index = pop_copy.loc[pop_copy['Year'] == year].index[0]
            if year not in result:
                result[year] = [int(pop_copy.loc[row_index, 'Population'])]
            else:
                result[year].append(int(pop_copy.loc[row_index, 'Population']))

    pop_modified = pd.DataFrame(result)
    return pop_modified


def column_name(df: pd.DataFrame) -> List[str]:
    """
    This takes a dataframe and returns the dataframe with modified column name.
    """
    column_list = df.columns.tolist()
    column_list[0] = 'County'
    column_list[1] = 'Year'
    column_list[2] = 'Population'
    return column_list


def clean_up_pop_df(filename: str, start_id: int, interval: int):
    """
    This function takes 1970-99 population dataset and returns cleaned up dataset
    with columns 'County', 'Year', 'Population'.
    """
    # import .xlsx file as .csv
    pop_df = get_df_pop(get_path(filename))
    pop_df.columns = column_name(pop_df)
    pop_df = pop_df.loc[start_id:, ['County', 'Year', 'Population']]

    # find rows with value in column 'County'
    na_series = pop_df.isna()['County']
    for i in range(start_id + 2, len(pop_df.axes[0])+start_id):
        if (na_series[i - 2] == False) and (na_series[i - 1] == False) and (na_series[i] == False):
            na_series[i] = True
        if (na_series[i - 1] == False) and (na_series[i] == False):
            na_series[i] = True
    na_series.iloc[-5:] = True
    false_rows = na_series[na_series == False]
    county_indexes = false_rows.index.tolist()

    # fill column 'County'
    for i in county_indexes:
        if type(pop_df.loc[i+1, 'County']) == str:
            pop_df.loc[i:i+interval, 'County'] = (str(pop_df.loc[i, 'County']) +
                                           " " + str(pop_df.loc[i+1, 'County'])).replace("  ", " ")

        else:
            pop_df.loc[i:i+interval, 'County'] = str(pop_df.loc[i, 'County'])

    # remove unnecessary rows
    pop_df = pop_df.dropna()
    index_lst = [[i for i in range(len(pop_df.axes[0]))]]
    pop_df = pop_df.set_index(index_lst)

    return pop_df


def get_geometry(mdf: pd.DataFrame) -> pd.DataFrame:
    """
    This functino takes mosquito dataset and converts it into GeoDataFrame.
    """
    # coordinates column
    coordinates = zip(mdf['decimalLongitude'], mdf['decimalLatitude'])
    mdf['coordinates'] = [
        Point(lon, lat) for lon, lat in coordinates
    ]

    # convert it to a geopandas
    mdf = gpd.GeoDataFrame(mdf, geometry='coordinates')
    return mdf


def get_map_ca():
    """
    This function returns US map GeoDataFrame.
    """
    gdf = gpd.read_file(get_path('cb_2018_us_county_20m.shp'))
    gdf = gdf[gdf['STATEFP'] == '06']
    return gdf


def filter_occurence_by_30_year(us_map: gpd.GeoDataFrame, occurence: pd.DataFrame, num: str):
    """
    Added a column in the given dataFrame that represents the (longitide, latitude) of the occurences
    in a given year.
    """
    coordinates = zip(occurence['decimalLongitude'], occurence['decimalLatitude'])
    occurence['coordinates' + num] = [Point(lon, lat) for lon, lat in coordinates]


def filter_us(occurence: pd.DataFrame):
    """
    Returns occurence dataset filtered by US
    """
    us_occurence = filter_ca(occurence)
    is_1904 = us_occurence["year"] == 2014
    new_df = us_occurence[is_1904]
    print("df:", new_df)
    # min_value = occurence["year"].min()
    # us_occurence = us_occurence.loc[:, ["species", "individualCount", "month", "year"]]
    # us_occurence = us_occurence["individualCount"].sum()
    # print(min_value)
    return us_occurence

# def merge_df(occurence1, occurence2):
#     result = occurence1.merge(occurence2, left_on="year",
#                             right_on="year", how="outer")
#     return result