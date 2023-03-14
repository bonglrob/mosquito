"""
CSE 163 Final Project
Kyoko Kurihara, Robert Bonglamphone, Christine Cai

This file implements test functions for mosquito.py.
"""
import mosquito as m
from typing import Any
import math
import pandas as pd


def assert_equals(expect: Any, actual: Any) -> None:
    """
    This function takes an expected value and an actual value,
    and checks if these two values are same.
    """
    if type(expect) in [int, float]:
        assert math.isclose(expect, actual),\
                f"Expected {expect}, but got {actual}"
    assert expect == actual, f"Expected {expect}, but got {actual}"


def in_column(column: Any, df: pd.DataFrame) -> None:
    """
    This function takes a column name and a DataFrame
    and checks if the column is in the DataFrame.
    """
    columns = df.columns.to_list()
    assert column in columns,\
        


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
    assert "Temp_Eureka" in actual.columns.to_list(),\
        "A column named 'Temp_Eureka' does not exist"
    assert "Prec_Los Angeles" in actual.columns.to_list(),\
        "A column named 'Prec_Eureka' does not exist"
    print("test_test_generate_city_df passed!")


def test_combine_pop_df() -> None:
    """
    This function tests if all population excel datasets are imported and
    combined to one DataFrame by combine_pop_df funciton from mosquito.py.
    """
    actual = m.combine_pop_df()
    # test type of result
    assert_equals(pd.DataFrame, type(actual))
    # test column name
    assert '1947' in actual.columns.to_list(),\
        "A column '1947' does not exist"
    # assert '19'


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
    test_generate_city_df()
    test_get_count_per_month(m.get_df_m(path3))
    print("all tests passed!")


if __name__ == '__main__':
    main()
