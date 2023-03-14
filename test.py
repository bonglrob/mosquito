"""
CSE 163 Final Project
Kyoko Kurihara, Robert Bonglamphone, Christine Cai

This file implements test functions for mosquito.py.
"""
import mosquito as m
from typing import Any
import math
import pandas as pd
import geopandas as gpd
import numpy as np


def assert_equals(expect: Any, actual: Any) -> None:
    """
    This function takes an expected value and an actual value,
    and checks if these two values are the same.
    """
    if type(expect) in [int, float]:
        assert math.isclose(expect, actual),\
                f"Expected {expect}, but got {actual}"
    assert expect == actual, f"Expected {expect}, but got {actual}"


def in_column(column: str, df: pd.DataFrame) -> None:
    """
    This function takes a column name and a DataFrame
    and checks if the column is in the DataFrame.
    """
    columns = df.columns.to_list()
    message = "A column '" + str(column) + "' does not exist"
    assert column in columns, message


def test_get_df_m(path: str) -> None:
    """
    This function tests if mosquito csv file is imported properly
    by get_df_m function from mosquito.py.
    """
    # test type of result
    actual = m.get_df_m(path)
    assert_equals(pd.DataFrame, type(actual))
    # test column name
    expect = ['species', 'countryCode', 'locality', 'stateProvince',
              'individualCount', 'decimalLatitude', 'decimalLongitude',
              'month', 'year']
    assert actual.columns.to_list() == expect,\
        f"Expected {expect}, but got {actual.columns.to_list()}"
    print("test_get_df_m passed for " + path + " passed!")


def test_generate_city_df() -> None:
    """
    This function tests if temperature and precipitation csv files are
    imported and cleaned up properly by generate_city_df function
    from mosquito.py.
    """
    actual = m.generate_city_df()
    # test type of result
    assert_equals(pd.DataFrame, type(actual))
    # test column name
    in_column("Temp_Eureka", actual)
    in_column("Prec_Los Angeles", actual)
    print("test_generate_city_df passed!")


def test_combine_pop_df() -> None:
    """
    This function tests if all population excel datasets are imported and
    combined to one DataFrame by combine_pop_df funciton from mosquito.py.
    """
    actual = m.combine_pop_df()
    # test type of result
    assert_equals(pd.DataFrame, type(actual))
    # test column name
    in_column('1947', actual)
    in_column('2022', actual)
    # test 'County' column
    assert len(actual['County']) == 58,\
        f"Expected {58}, but got {len(actual)['County']}"
    print("test_combine_pop_df_passed!")


def test_merge_all_data(df: pd.DataFrame):
    """
    This function tests if all collected data are combined properly to
    one GeoDataFrame by merge_all_data function from mosquito.py.
    """
    actual = m.merge_all_data(df)
    # test type of result
    assert_equals(pd.DataFrame, type(actual))
    # test column name
    in_column('population', actual)
    in_column('temperature', actual)
    in_column('year', actual)
    print("test_merge_all_data passed!")


def test_prediction(df: pd.DataFrame):
    """
    This function tests if the prediction function from mosquito.py
    returns a proper value.
    """
    mse1, gdf1 = m.prediction(df)
    # type
    assert_equals(np.float64, type(mse1))
    assert_equals(gpd.GeoDataFrame, type(gdf1))
    # column
    in_column('individualCount', gdf1)
    # when the depth is edited
    mse2, gdf2 = m.prediction(df, depth=100)
    assert mse1 != mse2 or gdf1 != gdf2,\
        "The depth chnge did not change result"
    # when return_features is True
    feature1 = m.prediction(df, return_features=True)
    # type
    assert_equals(pd.DataFrame, type(feature1))
    # column
    in_column('temperature', feature1)
    # when new_prediction is True
    mse3, gdf3, new3 = m.prediction(df, new_prediction=True,
                                    new_features=feature1)
    # type
    assert_equals(np.float64, type(mse3))
    assert_equals(gpd.GeoDataFrame, type(gdf3))
    assert_equals(gpd.GeoDataFrame, type(new3))
    # column
    in_column('individualCount', gdf3)
    in_column('individualCount', new3)
    assert gdf3.loc[0, 'decimalLongitude'] != new3.loc[0, 'decimalLongitude'],\
        "Prediciton on test set and prediction on new featrues are same"
    print("test_prediction passed!")


def test_decide_depth(df: pd.DataFrame, name: str):
    """
    This fucntion tests if the decide_depth function from mosquito.py returns
    an integer.
    """
    actual = m.decide_depth(df, name)
    # type
    assert_equals(int, type(actual))
    print("test_decide_depth for", name, "passed!")


def test_get_count_per_month(data: pd.DataFrame) -> pd.DataFrame:
    """
    Tests get_count_per_month to make sure it is accurately returning a
    DataFrame with all the correct monthly count values
    """
    culex_count = m.get_count_per_month(data)
    is_year_2018 = culex_count['year'] == 2018
    is_month_july = culex_count['month'] == 7
    culex_count = culex_count[is_year_2018 & is_month_july]['individualCount']

    assert_equals(200568, culex_count.item())


def main():
    path1 = m.get_path('Aedes_aegypti_occurrence.csv')
    path2 = m.get_path('Anopheles_quadrimaculatus_occurrence.csv')
    path3 = m.get_path('Culex_tarsalis_occurrence.csv')
    test_get_df_m(path1)
    test_get_df_m(path2)
    test_get_df_m(path3)
    mosquito1 = m.get_df_m(path1)
    mosquito3 = m.get_df_m(path3)
    test_generate_city_df()
    test_combine_pop_df()
    test_merge_all_data(mosquito1)
    test_merge_all_data(mosquito3)
    data1 = m.merge_all_data(mosquito1)
    data3 = m.merge_all_data(mosquito3)
    test_prediction(data1)
    test_prediction(data3)
    test_decide_depth(data1, "Aedes aegypti")
    test_decide_depth(data3, "Culex tarsalis")
    test_get_count_per_month(mosquito3)
    print("all tests passed!")


if __name__ == '__main__':
    main()
