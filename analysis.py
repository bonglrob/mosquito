import mosquito as m
import os
import pandas as pd
from shapely.geometry import Point
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from typing import List
from shapely.geometry import Point
import plotly.express as px
import calendar


def main() -> None:
    # read data
    mosquito1 = m.get_df_m(m.get_path('Aedes_aegypti_occurrence.csv'))
    mosquito2 = m.get_df_m(m.get_path('Anopheles_quadrimaculatus_occurrence.csv'))
    mosquito3 = m.get_df_m(m.get_path('Culex_tarsalis_occurrence.csv'))
    # question 1
    # prepare map of US
    # us_map = gpd.read_file(m.get_path('gz_2010_us_040_00_5m.json'))
    # us_map = us_map[(us_map['NAME'] != 'Alaska') & (us_map['NAME'] != 'Hawaii')]

    # 1904 - 2023
    # btn_04_33 = (mosquito1['year'] >= 1904) & (mosquito1['year'] <= 1933)
    # btn_34_63 = (mosquito1['year'] >= 1934) & (mosquito1['year'] <= 1963)
    # btn_64_93 = (mosquito1['year'] >= 1964) & (mosquito1['year'] <= 1993)
    # btn_94_23 = (mosquito1['year'] >= 1994) & (mosquito1['year'] <= 2023) &\
    #             (mosquito1['stateProvince'] != 'Hawaii')

    # occurrence_04_33 = mosquito1[btn_04_33]
    # occurrence_34_63 = mosquito1[btn_34_63]
    # occurrence_64_93 = mosquito1[btn_64_93]
    # occurrence_94_23 = mosquito1[btn_94_23]

    # m.filter_occurrence_by_30_year(occurrence_04_33, '1')
    # m.filter_occurrence_by_30_year(occurrence_34_63, '2')
    # m.filter_occurrence_by_30_year(occurrence_64_93, '3')
    # m.filter_occurrence_by_30_year(occurrence_94_23, '4')

    # fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, figsize=(20, 10))
    # us_map.plot(color='#EEEEEE', edgecolor='#FFFFFF', ax=ax1)
    # occurrence_points = gpd.GeoDataFrame(occurrence_04_33, geometry='coordinates1')
    # occurrence_points.plot(column='coordinates1', markersize=5, ax=ax1, vmin=0, vmax=1)
    # ax1.set_title('Occurrences of yellow fever mosquito in 1903-1933')

    # us_map.plot(color='#EEEEEE', edgecolor='#FFFFFF', ax=ax2)
    # occurrence_points = gpd.GeoDataFrame(occurrence_34_63, geometry='coordinates2')
    # occurrence_points.plot(column='coordinates2', markersize=5, ax=ax2, vmin=0, vmax=1)
    # ax2.set_title('Occurrences of yellow fever mosquito in 1934-1963')

    # us_map.plot(color='#EEEEEE', edgecolor='#FFFFFF', ax=ax3)
    # occurrence_points = gpd.GeoDataFrame(occurrence_64_93, geometry='coordinates3')
    # occurrence_points.plot(column='coordinates3', markersize=5, ax=ax3, vmin=0, vmax=1)
    # ax3.set_title('Occurrences of yellow fever mosquito in 1964-1993')

    # us_map.plot(color='#EEEEEE', edgecolor='#FFFFFF', ax=ax4)
    # occurrence_points = gpd.GeoDataFrame(occurrence_94_23, geometry='coordinates4')
    # occurrence_points.plot(column='coordinates4', markersize=5, ax=ax4, vmin=0, vmax=1)
    # ax4.set_title('Occurrences of yellow fever mosquito in 1994-2023')
    # plt.savefig('Occurrence_yellow_fever.png')

    # plt.show()
    # plt.close()


    # question 2
    aedes_species = mosquito1.copy()
    # species2 = mosquito2.copy()
    culex_species = mosquito3.copy() # https://www.wrbu.si.edu/vectorspecies/mosquitoes/tarsalis, Culex tarsalis Coquillett, 1896, usually tracked in

    aedes_species_count = aedes_species.groupby(['year', 'month', 'species'])['individualCount'].sum().reset_index()
    aedes_species_count['month'] = aedes_species_count['month'].apply(lambda x: calendar.month_name[int(x)])
    # species2_count = species2.groupby(['year', 'species']).agg({'individualCount': 'sum'}).reset_index()
    culex_species_count = culex_species.groupby(['year', 'month', 'species'])['individualCount'].sum().reset_index()
    culex_species_count['month'] = culex_species_count['month'].apply(lambda x: calendar.month_name[int(x)])

    # all_species = pd.concat([aedes_species_count, culex_species_count]).reset_index(drop=True)
    all_species = aedes_species_count.merge(culex_species_count, how='outer')

    # culex_species_count['individualCount'] = culex_species_count['individualCount'].diff()

    fig1 = px.bar(
        all_species,
        x='month',
        y='individualCount',
        color='species',
        title='Monthly Count by Species',
        labels={
        'individualCount': 'Count',
        'month': 'Month',
        'species': 'Species'
        },
        animation_frame='year'
        )

    fig1.update_layout(updatemenus=[dict(type='buttons', showactive=False,
                                   buttons=[dict(label='Play',
                                                 method='animate',
                                                 args=[None, {'frame': {'duration': 1000, 'redraw': True}, 'fromcurrent': True, 'transition': {'duration': 0}}])])])

    fig1.show()

    # question 3
    city_data = m.generate_city_df()
    # city_data.to_csv(m.get_path('city_data.csv'), index=False)
    pop_df = m.combine_pop_df()
    # pop_df.to_csv(m.get_path('pop_all.csv'), index=False)
    mosquito1_ca = m.filter_ca(mosquito1)
    mosquito2_ca = m.filter_ca(mosquito2)
    mosquito3_ca = m.filter_ca(mosquito3)

    m.merge_all_data(mosquito1)

    """
    # assign points to county !!

    # generate dataframe which has the total occurrence in the area,
    # species, year, month, area, areas temp, areas rainfall, areas temp columns, previous occurrence

    # use machine learning label: ocuurence, features other columns in the dataframe
    # Regression model

    # might use only July and August
    # testing: producing value
    # still have some problems...

    ca_map = m.get_map_ca()
    geom3_ca = m.ca_geomosquito(mosquito3)

    # fig, ax = plt.subplots(1, figsize=(15, 7))
    # ca_map.plot(ax=ax, color='#EEEEEE', edgecolor='#FFFFFF')

    # county_with_m = gpd.sjoin(ca_map, geom3_ca, how='inner', op='intersects')
    # county_with_m.plot(ax=ax)
    # geom3_ca.plot(color='red', markersize=2, ax=ax)

    # plt.show()
    plt.close()
    """


if __name__ == '__main__':
    main()
