"""
CSE 163 Final Project
Kyoko Kurihara, Robert Bonglamphone, Christine Cai

This file contains functions for analyzing mosquito occurrence.
It is not a runnable code.

1. The code converts three mosquito occurrence CSV datasets to a GeoDataFrame
   with latitude and longitude information.

2. In order to write a GeoDataFrame
   for machine learning purposes, the code changes the formats of population,
   temperature, and precipitation files, and combines them all
   with the mosquito occurrence GeoPandaDataFrame.
    - For the population datasets, it writes a new DataFrame by cleaning up
      seven CSV files.
    - For temperature and precipitation datasets, it writes
      a new DataFrame with 12 CSV files.

3. The machine learning part uses:
    - features: 'population', 'temperature', 'precipitation', 'location', etc.
    - lables: mosquito occurrence 'latitude', 'longitude', 'individual counts'

   It trains a Random Forest regressor model.

4. The prediction results are then plotted for each mosquito species.
"""
import os
from shapely.geometry import Point
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import plotly.express as px
import calendar
from typing import List
from typing import Any
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def get_path(filename: str) -> str:
    """
    This function takes a file name (str) and returns the file path (str).
    """
    return os.path.join('./dataset', filename)


def to_int(x: Any) -> Any:
    """
    This function takes a number and converts it into an integer
    if it is not Nan.
    """
    if pd.notna(x):
        return int(str(x).split('.')[0])
    else:
        return x


def get_df_m(file_path: str) -> pd.DataFrame:
    """
    This function takes a file path (str) of mosquito occurrence
    dataset and returns it as a DataFrame.
    """
    data = pd.read_csv(file_path, delimiter='\t', low_memory=False)
    # select columns
    data = data.loc[data['countryCode'] == 'US',
                    ['species', 'countryCode', 'locality',
                     'stateProvince', 'individualCount',
                     'decimalLatitude', 'decimalLongitude',
                     'month', 'year']
                    ]
    # edit DataFrame
    data = data.dropna(subset=['year'])
    # fill Nan in "individual count" with 1
    data["individualCount"] = data["individualCount"].fillna(1)
    data['month'] = data['month'].apply(to_int)
    data['year'] = data['year'].apply(to_int)
    return data


def filter_ca(df: pd.DataFrame) -> pd.DataFrame | None:
    """
    This function takes a mosquito occurrence DataFrame and returns
    only California data.
    """
    # filter rows
    mask = df['stateProvince'] == 'California'
    result = df[mask]
    # if there is no occurrence in California
    if result.empty:
        print("No occurrence in California")
        return None
    else:
        return result


def get_geometry(mdf: pd.DataFrame) -> pd.DataFrame | None:
    """
    This function takes mosquito dataset and converts it into GeoDataFrame.
    """
    if mdf is None:
        return None
    else:
        # coordinates column
        mdf = mdf.copy()
        mdf.loc[:, 'coordinates'] = \
            mdf.apply(lambda row: Point(row['decimalLongitude'],
                                        row['decimalLatitude']), axis=1)

        # convert it to a geopandas
        mdf = gpd.GeoDataFrame(mdf, geometry='coordinates')
        return mdf


def ca_geomosquito(df: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    This function takes a DataFrame and returns filtered GeoDataFrame
    with California data.
    """
    # if DataFrame is empty
    if df is None:
        print("No occurrence in California")
    else:
        filtered = filter_ca(df)
        result = get_geometry(filtered)
        return result


def get_map_ca() -> gpd.GeoDataFrame:
    """
    This function returns US map GeoDataFrame.
    """
    gdf = gpd.read_file(get_path('cb_2018_us_county_20m.shp'))
    gdf = gdf[gdf['STATEFP'] == '06']
    return gdf


def ca_occurrence(mosquito: pd.DataFrame) -> gpd.GeoDataFrame:
    # import California map
    ca_map = get_map_ca()
    # read occurrence dataset
    mosquito_ca = ca_geomosquito(mosquito)
    mosquito_ca.crs = "EPSG:4269"
    # merge datasets
    merged = gpd.sjoin(ca_map, mosquito_ca, how='inner',
                       predicate='intersects')
    columns_to_drop = ['STATEFP', 'COUNTYNS', 'AFFGEOID', 'GEOID']
    merged = merged.drop(columns_to_drop, axis=1)
    return merged


def get_df_t(file_path: str) -> pd.DataFrame:
    """
    This function takes a file path (str) of temperature dataset
    and converts it to a DataFrame.
    """
    data = pd.read_csv(file_path)
    data.columns = ["Date", "Temp", "Anomaly"]
    data = data.loc[4:, "Date":"Temp"]
    data["Temp"] = data["Temp"].astype(float)
    return data


def get_df_p(file_path: str) -> pd.DataFrame:
    """
    This function takes a file path (str) of one of precipitation datasets
    and converts it to a DataFrame.
    """
    data = pd.read_csv(file_path)
    data.columns = ["Date", "Prec", "Anomaly"]
    data = data.loc[4:, "Date":"Prec"]
    data["Prec"] = data["Prec"].astype(float)
    return data


def generate_city_df() -> pd.DataFrame:
    """
    This function returns a DataFrame which combines all city tempereature
    and precipitation DataFrames.
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
    result.to_csv('./results/city_data.csv', index=False)
    return result


def get_df_pop(file_name: str, sheetname=None) -> pd.DataFrame:
    """
    This function takes a file name of population datasets and converts
    it into a DataFrame.
    """
    if sheetname is None:
        data = pd.DataFrame(pd.read_excel(file_name))
    if sheetname is not None:
        data = pd.DataFrame(pd.read_excel(file_name, sheet_name=sheetname))
    return data


def pop50() -> pd.DataFrame:
    """
    This function reads 'popca_4769.xlsx' and returns its cleaned-up DataFrame.
    """
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


def column_name(df: pd.DataFrame) -> List[str]:
    """
    This takes a DataFrame and returns the DataFrame with modified column name.
    """
    column_list = df.columns.tolist()
    column_list[0] = 'County'
    column_list[1] = 'Year'
    column_list[2] = 'Population'
    return column_list


def clean_up_pop_df(filename: str, start_id: int, interval: int,
                    sheetname: Any = None):
    """
    This function takes 1970-2022 population dataset and cleans up the dataset
    so that is has columns 'County', 'Year', and 'Population'.
    """
    # import .xlsx file as .csv
    pop_df = get_df_pop(get_path(filename), sheetname=sheetname)
    pop_df.columns = column_name(pop_df)
    pop_df = pop_df.loc[start_id:, ['County', 'Year', 'Population']]

    # find rows with value in column 'County'
    na_edit = pop_df.fillna('blank')['County'].copy()

    for i in range(start_id + 2, len(pop_df.axes[0])+start_id):
        if (na_edit[i - 2] != 'blank') and \
           (na_edit[i - 1] != 'blank') and \
           (na_edit[i] != 'blank'):
            na_edit[i] = 'blank'
        if (na_edit[i - 1] != 'blank') and (na_edit[i] != 'blank'):
            na_edit[i] = 'blank'
    na_edit.iloc[-5:] = 'blank'
    county_indexes = na_edit[na_edit != 'blank'].index.to_list()

    # fill column 'County'
    for i in county_indexes:
        if type(pop_df.loc[i+1, 'County']) == str:
            pop_df.loc[i:i+interval, 'County'] = \
                (str(pop_df.loc[i, 'County']) + " " +
                 str(pop_df.loc[i+1, 'County'])).replace("  ", " ")

        else:
            pop_df.loc[i:i+interval, 'County'] = str(pop_df.loc[i, 'County'])

    # remove unnecessary rows
    pop_df = pop_df.dropna()
    index_lst = [[i for i in range(len(pop_df.axes[0]))]]
    pop_df = pop_df.set_index(index_lst)

    return pop_df


def change_style(pop: pd.DataFrame) -> pd.DataFrame:
    """
    This function takes a population DataFrame and changes its styles so that
    it has county name on the 1st column and years as columns name.
    """
    result = {'County': []}

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


def combine_pop_df() -> pd.DataFrame:
    """
    This funciton combines all population DataFrames and cleans it up.
    """
    # read files
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

    # merge DataFrames
    pop_all = pop_50.merge(pop_70, left_on='County', right_on='County',
                           how='left')
    pop_all = pop_all.merge(pop_80, left_on='County', right_on='County',
                            how='left')
    pop_all = pop_all.merge(pop_90, left_on='County', right_on='County',
                            how='left')
    pop_all = pop_all.merge(pop_00, left_on='County', right_on='County',
                            how='left')
    pop_all = pop_all.merge(pop_10, left_on='County', right_on='County',
                            how='left')
    pop_all = pop_all.merge(pop_20, left_on='County', right_on='County',
                            how='left')

    # edit columns
    pop_all.columns = pop_all.columns.astype(str)
    pop_all.to_csv('./results/pop_all.csv', index=False)
    return pop_all


def get_capital(county: str) -> str:
    """
    This funciton takes a county name (str) and returns the capital city
    that the county belongs to.
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
            if capital == 'Humboldt':
                return 'Eureka'
            return capital
    return None


def add_0(num: float) -> str:
    """
    This function takes a number between 1 to 12 and
    returns it in a format of '0X' or '1X'.
    """
    if num < 10:
        result = '0' + str(int(num))
    else:
        result = str(int(num))
    return result


def merge_all_data(mosquito: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    This function takes mosquito occurrence Dataframe and merges it
    with population, temperature, precipitation datasets.
    The returned GeoDataFrame will be used for machine learning.
    """
    # prepare all required datasets
    occurrence = ca_occurrence(mosquito)
    city_df = generate_city_df()
    city_df = city_df.reset_index(drop=True)
    pop_df = combine_pop_df()

    occurrence = occurrence[(occurrence['year'] < 2023) &
                            (occurrence['year'] > 1946)]
    occurrence = occurrence.reset_index(drop=True)

    for i in occurrence.index.tolist():
        # population
        year = str(int(occurrence.loc[i, 'year']))
        county = occurrence.loc[i, 'NAME']
        occurrence.loc[i, 'population'] = \
            pop_df.loc[pop_df.loc[pop_df['County'] == county].index[0], year]

        # temperature & precipitation
        capital = get_capital(county)
        month = occurrence.loc[i, 'month']
        yyyymm = str(int(year))+add_0(month)
        temp = 'Temp_' + capital
        prec = 'Prec_' + capital
        occurrence.loc[i, 'temperature'] = \
            float(city_df.loc[city_df.loc[city_df['Date'] == yyyymm].index[0],
                              temp])
        occurrence.loc[i, 'precipitation'] = \
            float(city_df.loc[city_df.loc[city_df['Date'] == yyyymm].index[0],
                              prec])
    occurrence = occurrence.drop(columns=['COUNTYFP', 'index_right', 'species',
                                          'countryCode', 'locality',
                                          'stateProvince', 'geometry'])
    occurrence = occurrence.dropna()
    return occurrence


def prediction(data: gpd.GeoDataFrame, depth: Any = None,
               new_prediction: bool = False, new_features: Any = None,
               random: int = 163, return_features: bool = False) -> Any:
    """
    This function takes a GeoDataFrame with mosquito occurrence, population,
    temperature, precipitation etc., for each county in California,
    and predicts latitude, longitude, and individual count of
    mosquito occurences by training a Random Forest regressor model.
    """
    # select features and labels
    features = data.drop(columns=['decimalLongitude', 'decimalLatitude',
                                  'individualCount', 'NAME'])
    # return new features
    if return_features:
        return features
    labels = data[['decimalLongitude', 'decimalLatitude', 'individualCount']]

    # split data into training and testing datasets
    feature_train, feature_test, label_train, label_test = \
        train_test_split(features, labels, test_size=0.2, random_state=random)

    # creaate a Random Forest regressor model
    rf = RandomForestRegressor(n_estimators=depth, random_state=random)

    # train the model
    rf.fit(feature_train, label_train)

    # predict on testing set
    y_pred = rf.predict(feature_test)

    # compute the mean squared error
    mse = mean_squared_error(label_test, y_pred)

    # create a new GeoDataFrame with the predicted longitude and latitude
    predictions = pd.DataFrame(y_pred, columns=['decimalLongitude',
                                                'decimalLatitude',
                                                'individualCount'])
    predictions_gdf = get_geometry(predictions)
    predictions_gdf.set_crs("EPSG:4269", inplace=True)

    # make predictions on the new features
    if new_prediction:
        new_pred = rf.predict(new_features)
        new = pd.DataFrame(new_pred, columns=['decimalLongitude',
                                              'decimalLatitude',
                                              'individualCount'])
        new_gdf = get_geometry(new)
        new_gdf.set_crs("EPSG:4269", inplace=True)
        return mse, predictions_gdf, new_gdf

    return mse, predictions_gdf


def decide_depth(df: pd.DataFrame, mosquito: str, random: int = 163,) -> int:
    """
    This function takes a mosquito DataFrame and returns the tree depth
    that can minimize the error.
    """
    randoms = [i for i in range(1, 30)]
    mses = []
    # plot depth vs error
    for num in randoms:
        mse, gdf = prediction(df, num, random=random)
        mses.append(mse)
    plt.plot(randoms, mses, 'ko-')
    plt.xlabel("Random Forest Depth")
    plt.ylabel("Mean Squared Error")
    plt.title("Random Forest Depth vs Mean Squared Error (" + mosquito + ")")
    plt.savefig('./results/error_' + mosquito + '.png')
    plt.show()
    plt.close()
    return randoms[mses.index(min(mses))]


def plot_prediction(gdf: gpd.GeoDataFrame, title: str) -> None:
    """
    This function takes a GeoDataFrame of mosquito occurence and plots it
    on a California map.
    """
    fig, ax = plt.subplots(1)
    # plot a CA map
    ca_map = get_map_ca()
    ca_map.plot(ax=ax, color='#EEEEEE', edgecolor='#FFFFFF')
    # plot the predicted points on a map
    sizes = gdf['individualCount']
    gdf.plot(ax=ax, column='individualCount', cmap='coolwarm',
             markersize=sizes)
    plt.title(title)
    plt.savefig('./results/' + title + '.png')
    plt.show()
    plt.close()


def filter_occurrence_by_30_year(occurrence: pd.DataFrame,
                                 num: str) -> pd.DataFrame:
    """
    Added a column in the given dataFrame that represents the (longitide,
    latitude) of the occurrences in a given year.
    """
    occurrence = occurrence.copy()
    coordinates = zip(occurrence['decimalLongitude'],
                      occurrence['decimalLatitude'])
    occurrence['coordinates' + num] = \
        [Point(lon, lat) for lon, lat in coordinates]
    return occurrence


def get_count_per_month(data: pd.DataFrame) -> pd.DataFrame:
    """
    For the given mosquito species data, return the data with a sum of
    occurence per month for each year
    """
    return data.groupby(['year', 'month', 'species'])['individualCount'] \
               .sum() \
               .reset_index()


def filter_years(data: pd.DataFrame) -> pd.DataFrame:
    """
    For the given mosquito species data, return filtered data of counts between
    year 2002 and 2022
    """
    is_years_2002_2022 = (data['year'] >= 2002) & (data['year'] <= 2022)
    return data[is_years_2002_2022]


def add_month_name(data: pd.DataFrame) -> pd.DataFrame:
    """
    For given mosquito species data, return data with column of month name for
    each corresponding number (e.g. 1 -> January) to increase readability for
    plots
    """
    data['month_name'] = data['month'].apply(lambda x:
                                             calendar.month_name[int(x)])
    return data


def merge_all_species_data(aedes_data: pd.DataFrame,
                           anopheles_data: pd.DataFrame,
                           culex_data: pd.DataFrame) -> pd.DataFrame:
    """
    For each given mosquito species data, return the combined version of all
    monthly counts
    """

    aedes_anopheles_merge_df = pd.merge(aedes_data,
                                        anopheles_data,
                                        how='outer')

    all_species_count = pd.merge(aedes_anopheles_merge_df,
                                 culex_data,
                                 how='outer')

    return all_species_count


def plot_species(data: pd.DataFrame) -> None:
    """
    For given data on all mosquito species, opens a browser page to plot a
    line graph of each species and their count per month for 2002.
    The year can be adjusted up to 2022 using the slider input.
    Each species can also be toggled on/off if you click on them in the legend
    """
    fig_species = px.line(
        data,
        x='month',
        y='individualCount',
        color='species',
        title='Monthly Count of Mosquitoes in the US by Species',
        labels={
            'individualCount': 'Count',
            'month': 'Month',
            'species': 'Species'
        },
        animation_frame='year'
    )

    fig_species.update_layout(
        yaxis=dict(range=[0, data['individualCount'].max()]),
    )

    fig_species.update_xaxes(ticktext=data['month_name'],
                             tickvals=data['month'])

    fig_species.show()


def plot_aedes(data: pd.DataFrame) -> None:
    """
    For given data on aedes species, opens a browser page to plot a
    line graph of their count per month of every year from 2002 t0 2022.
    Each year can be toggled on/off by clicking on them in the legend.
    """
    fig_aedes = px.line(
        data,
        x='month',
        y='individualCount',
        color='year',
        title='Monthly Counts of Aedes Aegypti from 2002 to 2022',
        labels={
            'individualCount': 'Count',
            'month': 'Month',
            'color': 'Year'
        }
    )

    fig_aedes.update_xaxes(ticktext=data['month_name'], tickvals=data['month'])
    fig_aedes.show()
