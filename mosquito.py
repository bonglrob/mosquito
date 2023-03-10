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
from typing import Any
from shapely.geometry import Point


def get_path(filename: str) -> str:
    """
    This function takes a file name and returns the file path.
    """
    return os.path.join('./dataset', filename)


def to_int(x: Any):
    if pd.notna(x):
        return int(str(x).split('.')[0])
    else:
        return x


def get_df_m(file_path: str) -> pd.DataFrame:
    """
    This function takes a file name (str) of one of mosquito occurrence
    datasets and returns the dataframe.
    """
    # is there any other way "low_memory=False"
    data = pd.read_csv(file_path, delimiter='\t', low_memory=False)
    # select columns
    data = data.loc[data['countryCode'] == 'US',
                    ['species', 'countryCode', 'locality',
                     'stateProvince', 'individualCount',
                     'decimalLatitude', 'decimalLongitude',
                     'month', 'year']
                    ]
    data = data.dropna(subset=['year'])
    data["individualCount"] = data["individualCount"].fillna(1)
    data['month'] = data['month'].apply(to_int)
    data['year'] = data['year'].apply(to_int)
    return data


def filter_ca(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function takes a mosquito occurrence
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
    cities = ["Eureka", "Fresno", "Los Angeles", "Sacramento",
              "San Diego", "San Francisco"]
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
    result = result.dropna()
    return result


def get_df_pop(file_name, sheetname=None):
    if sheetname is None:
        data = pd.DataFrame(pd.read_excel(file_name))
    if sheetname is not None:
        data = pd.DataFrame(pd.read_excel(file_name, sheet_name=sheetname))
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
    # TODO: one of file divided by hand, which needs to be fixed quickly
    pop70 = clean_up_pop_df('popca_7080.xlsx', 15, 9)
    pop80 = clean_up_pop_df('popca_8090.xlsx', 15, 9)
    pop90 = clean_up_pop_df('popca_9000.xlsx', 15, 9)
    pop00 = clean_up_pop_df('popca_0010.xlsx', 20, 10)
    pop00 = pop00.drop(pop00[pop00['Year'] == 1999].index)
    pop10 = clean_up_pop_df('popca_1021.xlsx', 20, 10)
    pop10 = pop10.loc[pop10['Year'] != 'Census 2010']
    pop10['Year'] = pop10['Year'].replace('Apr-Jun 2010', 2010)
    pop20 = clean_up_pop_df('popca_2022.xlsx', 9, 4, sheetname=1)
    pop20 = pop20.loc[pop20['Year'] != 'Census 2020']
    pop20['Year'] = pop20['Year'].replace('Apr-Jun 2020', 2020)

    pop_50 = pop50()
    pop_70 = change_style(pop70)
    pop_80 = change_style(pop80)
    pop_90 = change_style(pop90)
    pop_00 = change_style(pop00)
    pop_10 = change_style(pop10)
    pop_20 = change_style(pop20)

    # merge
    pop_all = pop_50.merge(pop_70, left_on='County', right_on='County', how='left')
    pop_all = pop_all.merge(pop_80, left_on='County', right_on='County', how='left')
    pop_all = pop_all.merge(pop_90, left_on='County', right_on='County', how='left')
    pop_all = pop_all.merge(pop_00, left_on='County', right_on='County', how='left')
    pop_all = pop_all.merge(pop_10, left_on='County', right_on='County', how='left')
    pop_all = pop_all.merge(pop_20, left_on='County', right_on='County', how='left')

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


def clean_up_pop_df(filename: str, start_id: int, interval: int, sheetname=None):
    """
    This function takes 1970-99 population dataset and returns cleaned up dataset
    with columns 'County', 'Year', 'Population'.
    """
    # import .xlsx file as .csv
    pop_df = get_df_pop(get_path(filename), sheetname=sheetname)
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


def ca_geomosquito(df: pd.DataFrame):
    """
    This function takes DataFrame and returns filtered GeoDataFrame.
    """
    result = get_geometry(filter_ca(df))
    return result


def ca_occurrence(mosquito: pd.DataFrame):
    ca_map = get_map_ca()
    mosquito_ca = ca_geomosquito(mosquito)
    merged = gpd.sjoin(ca_map, mosquito_ca, how='inner', op='intersects')
    columns_to_drop = ['STATEFP', 'COUNTYNS', 'AFFGEOID', 'GEOID']
    merged = merged.drop(columns_to_drop, axis=1)
    return merged


def get_capital(county: str) -> str:
    """
    This funciton takes a county name (str) and returns the capital city that the county belongs to.
    """
    gp_eureka = ["Humboldt", "Del Norte", "Lake", "Mendocino", "Modoc",
                 "Shasta", "Siskiyou", "Tahama", "Trinity"]
    gp_fresno = ["Fresno", "Inyo", "Kern", "Kings", "Madera", "Mariposa",
                 "Merced", "Mono", "Tulare", "Tuolumne"]
    gp_sacramento = ["Sacramento", "Alpine", "Amador", "Butte", "Calaveras",
                     "Colusa", "El Dorado", "Glenn", "Lassen", "Nevada",
                     "Placer", "Plumas", "Sutter", "Sierra", "Stanislaus",
                     "Solano", "Yolo", "Yuba"]
    gp_sd = ["San Diego", "Imperial"]
    gp_sf = ["San Francisco", "Alameda", "Contra Costa", "Marin",
             "Napa", "San Mateo", "Santa Clara", "Santa Cruz", "Sonoma"]
    gp_la = ["Los Angeles", "Monterey", "Orange", "Riverside", "San Benito",
             "San Bernardino", "San Joaquin", "Santa Barbara",
             "San Luis Obispo", "Ventura"]
    groups = [gp_eureka, gp_fresno, gp_sacramento, gp_sd, gp_sf, gp_la]
    for group in groups:
        if county in group:
            capital = group[0]
            if capital == 'Euraka':
                return 'Humboldt'
            return capital


def add_0(num: float) -> str:
    """
    This function takes a number between 1 to 12 and returns it in a format of 'XX'.
    """
    if num < 10:
        result = '0' + str(int(num))
    else:
        result = str(int(num))
    return result


def merge_all_data(mosquito: pd.DataFrame):
    occurrence = ca_occurrence(mosquito)
    city_df = generate_city_df()
    city_df = city_df.reset_index(drop=True)
    pop_df = combine_pop_df()
    # print(len(gp_eureka) + len(gp_fresno) + len(gp_sacramento) + len(gp_sd) + len(gp_sf) + len(gp_la), "Counties")
    # print(len(set(gp_eureka + gp_fresno + gp_sacramento + gp_sd + gp_sf + gp_la)), "Counties")

    occurrence = occurrence[occurrence['year'] != 2023.0]
    occurrence = occurrence.reset_index(drop=True)

    for i in occurrence.index.tolist():
        # population
        year = occurrence.loc[i, 'year']
        county = occurrence.loc[i, 'NAME']
        occurrence.loc[i, 'population'] = pop_df.loc[pop_df.loc[pop_df['County'] == county].index[0], int(year)]

        # temperature & precipitation
        capital = get_capital(county)
        month = occurrence.loc[i, 'month']
        yyyymm = str(int(year))+add_0(month)
        temp = 'Temp_' + capital
        prec = 'Prec_' + capital
        occurrence.loc[i, 'temperature'] = city_df.loc[city_df.loc[city_df['Date'] == yyyymm].index[0], temp]
        occurrence.loc[i, 'precipitation'] = city_df.loc[city_df.loc[city_df['Date'] == yyyymm].index[0], prec]
    return occurrence


def prediction(data):
    """
    This function TBD
    """
    # import necessary libraries
    from sklearn.ensemble import RandomForestRegressor
    # from sklearn.preprocessing import OneHotEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    import geopandas as gpd
    import pandas as pd

    # encoder = OneHotEncoder(sparse=False)
    # encoded_features = encoder.fit_transform(data[['NAME']])
    dummies = pd.get_dummies(data['NAME'])
    data = pd.concat([data, dummies], axis=1)
    print(data.columns)
    data.to_csv('prediction.csv', index=False)
    data = data.drop(columns=[ 'NAME', 'index_right','species', 'countryCode', 'locality', 'stateProvince'])
    # # select the features and labels
    features = data.drop(columns=['decimalLongitude', 'decimalLatitude', 'individualCount', 'geometry'])
    labels = data[['decimalLongitude', 'decimalLatitude', 'individualCount']]

    # # split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # # create a Random Forest regressor model
    rf = RandomForestRegressor(n_estimators=100, random_state=42)

    # # train the model
    rf.fit(X_train, y_train)

    # # make predictions on the testing set
    y_pred = rf.predict(X_test)

    # # calculate the mean squared error
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse)

    # # create a new GeoDataFrame with the predicted longitude and latitude
    predictions = pd.DataFrame(y_pred, columns=['decimalLongitude', 'decimalLatitude', 'individualCount'])
    geometry = gpd.points_from_xy(predictions['decimalLongitude'], predictions['decimalLatitude'])
    predictions_gdf = gpd.GeoDataFrame(predictions, geometry=geometry)

    # # plot the predicted points on a map
    ax = data.plot(figsize=(10, 10), alpha=0.5, edgecolor='k')
    predictions_gdf.plot(ax=ax, color='r', markersize=20)
    plt.show()


def filter_occurence_by_30_year(us_map: gpd.GeoDataFrame, occurence: pd.DataFrame, num: str):
    """
    Added a column in the given dataFrame that represents the (longitide, latitude) of the occurrences
    in a given year.
    """
    coordinates = zip(occurrence['decimalLongitude'], occurrence['decimalLatitude'])
    occurrence['coordinates' + num] = [Point(lon, lat) for lon, lat in coordinates]


# def filter_us(occurence: pd.DataFrame):
#     """
#     Returns occurence dataset filtered by US
#     """
#     us_occurence = filter_ca(occurence)
#     is_1904 = us_occurence["year"] == 2014
#     new_df = us_occurence[is_1904]
#     print("df:", new_df)
#     # min_value = occurence["year"].min()
#     # us_occurence = us_occurence.loc[:, ["species", "individualCount", "month", "year"]]
#     # us_occurence = us_occurence["individualCount"].sum()
#     # print(min_value)
#     return us_occurence
