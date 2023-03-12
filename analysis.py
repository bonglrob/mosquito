import mosquito as m
# import os
import pandas as pd
from shapely.geometry import Point
# import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
# from typing import List
import plotly.express as px
import calendar


def main() -> None:
    # read data
    mosquito1 = m.get_df_m(m.get_path('Aedes_aegypti_occurrence.csv'))
    mosquito2 = m.get_df_m(m.get_path('Anopheles_quadrimaculatus_occurrence.csv'))
    mosquito3 = m.get_df_m(m.get_path('Culex_tarsalis_occurrence.csv'))

    """
    # question 1
    # prepare map of US
    # us_map = gpd.read_file(m.get_path('gz_2010_us_040_00_5m.json'))
    # us_map = us_map[(us_map['NAME'] != 'Alaska') & (us_map['NAME'] != 'Hawaii')]

    1904 - 2023
    btn_04_33 = (mosquito1['year'] >= 1904) & (mosquito1['year'] <= 1933)
    btn_34_63 = (mosquito1['year'] >= 1934) & (mosquito1['year'] <= 1963)
    btn_64_93 = (mosquito1['year'] >= 1964) & (mosquito1['year'] <= 1993)
    btn_94_23 = (mosquito1['year'] >= 1994) & (mosquito1['year'] <= 2023) &\
                (mosquito1['stateProvince'] != 'Hawaii')

    occurrence_04_33 = mosquito1[btn_04_33]
    occurrence_34_63 = mosquito1[btn_34_63]
    occurrence_64_93 = mosquito1[btn_64_93]
    occurrence_94_23 = mosquito1[btn_94_23]

    m.filter_occurrence_by_30_year(occurrence_04_33, '1')
    m.filter_occurrence_by_30_year(occurrence_34_63, '2')
    m.filter_occurrence_by_30_year(occurrence_64_93, '3')
    m.filter_occurrence_by_30_year(occurrence_94_23, '4')

    fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, figsize=(20, 10))
    us_map.plot(color='#EEEEEE', edgecolor='#FFFFFF', ax=ax1)
    occurrence_points = gpd.GeoDataFrame(occurrence_04_33, geometry='coordinates1')
    occurrence_points.plot(column='coordinates1', markersize=5, ax=ax1, vmin=0, vmax=1)
    ax1.set_title('Occurrences of yellow fever mosquito in 1903-1933')

    us_map.plot(color='#EEEEEE', edgecolor='#FFFFFF', ax=ax2)
    occurrence_points = gpd.GeoDataFrame(occurrence_34_63, geometry='coordinates2')
    occurrence_points.plot(column='coordinates2', markersize=5, ax=ax2, vmin=0, vmax=1)
    ax2.set_title('Occurrences of yellow fever mosquito in 1934-1963')

    us_map.plot(color='#EEEEEE', edgecolor='#FFFFFF', ax=ax3)
    occurrence_points = gpd.GeoDataFrame(occurrence_64_93, geometry='coordinates3')
    occurrence_points.plot(column='coordinates3', markersize=5, ax=ax3, vmin=0, vmax=1)
    ax3.set_title('Occurrences of yellow fever mosquito in 1964-1993')

    us_map.plot(color='#EEEEEE', edgecolor='#FFFFFF', ax=ax4)
    occurrence_points = gpd.GeoDataFrame(occurrence_94_23, geometry='coordinates4')
    occurrence_points.plot(column='coordinates4', markersize=5, ax=ax4, vmin=0, vmax=1)
    ax4.set_title('Occurrences of yellow fever mosquito in 1994-2023')
    plt.savefig('Occurrence_yellow_fever.png')

    # plt.show()
    plt.close()
    """

    # question 2 : Has there been a change of pattern for mosquitoes to thrive in certain months?

    # Copy aedes_species_occurence data
    aedes_species = mosquito1.copy()
    anopheles_species = mosquito2.copy()
    culex_species = mosquito3.copy()

    # Groupby count per month
    aedes_species_count = aedes_species.groupby(['year', 'month', 'species'])['individualCount'].sum().reset_index()
    anopheles_species_count = anopheles_species.groupby(['year', 'month', 'species'])['individualCount'].sum().reset_index()
    culex_species_count = culex_species.groupby(['year', 'month', 'species'])['individualCount'].sum().reset_index()

    # Filter for 2002 - 2022:
    is_years_2002_2022 = (aedes_species_count['year'] >= 2002) & (aedes_species_count['year'] <= 2022)
    aedes_species_count = aedes_species_count[is_years_2002_2022]
    is_years_2002_2022_2 = (anopheles_species_count['year'] >= 2002) & (anopheles_species_count['year'] <= 2022)
    anopheles_species_count = anopheles_species_count[is_years_2002_2022_2]
    is_years_2002_2022_3 = (culex_species_count['year'] >= 2002) & (culex_species_count['year'] <= 2022)
    culex_species_count = culex_species_count[is_years_2002_2022_3]

    # Change month into readable format for plot
    aedes_species_count['month_name'] = aedes_species_count['month'].apply(lambda x: calendar.month_name[int(x)])
    anopheles_species_count['month_name'] = anopheles_species_count['month'].apply(lambda x: calendar.month_name[int(x)])
    culex_species_count['month_name'] = anopheles_species_count['month'].apply(lambda x: calendar.month_name[int(x)])

    # Merge
    aedes_anopheles_merge_df = pd.merge(aedes_species_count, anopheles_species_count, how='outer')
    all_species_count = pd.merge(aedes_anopheles_merge_df, culex_species_count, how='outer')

    fig1 = px.line(
        all_species_count,
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

    fig1.update_layout(
        yaxis=dict(range=[0, all_species_count['individualCount'].max()]),
        updatemenus=[dict(type='buttons', showactive=False,
                                   buttons=[dict(label='Play',
                                                 method='animate',
                                                 args=[None, {'frame': {'duration': 1000, 'redraw': True}, 'fromcurrent': True, 'transition': {'duration': 0}}])])]
    )

    fig1.update_xaxes(ticktext=aedes_species_count['month_name'], tickvals=aedes_species_count['month'])


    fig1.show()

    # question 3
    # TODO: Improve model
    # Aedes aegypti
    # mosquito1 = m.get_df_m(m.get_path('Aedes_aegypti_occurrence.csv'))
    mosquito1_2022 = mosquito1[mosquito1['year'] == 2022]
    past1 = m.ca_geomosquito(mosquito1_2022)
    data1 = m.merge_all_data(mosquito1)

    min_error1 = m.decide_depth(data1)
    # decide new features
    new_features1 = m.prediction(data1, return_featues=True)
    new_features1['population'] = new_features1['population'] / 5
    new_features1['year'] = new_features1['year'] + 1000
    new_features1['temperature'] = new_features1['temperature'] - 50
    new_features1['precipitation'] = new_features1['precipitation'] + 1

    mse1, prediction1, new1 = m.prediction(data1, min_error1, new_prediction=True, new_features=new_features1)
    print("Mean Squared Error:", mse1)
    m.plot_prediction(past1, "Aedes aegypti occurrence in California in 2022")
    m.plot_prediction(prediction1, "Aedes aegypti occurrence in California by prediction on test features")
    m.plot_prediction(new1, "Aedes aegypti occurrence in California in a given condition")

    # Anopheles quadrimaculatus
    mosquito2_2022 = mosquito2[mosquito2['year'] == 2022]
    past2 = m.ca_geomosquito(mosquito2_2022)
    data2 = m.merge_all_data(mosquito2)
    past = m.ca_geomosquito(mosquito2)
    m.plot_prediction(past, "No occurrence of Anopheles quadrimaculatus in California")

    # min_error2 = m.decide_depth(data2)
    # # decide new features
    # new_features2 = m.prediction(data2, return_featues=True)
    # new_features2['population'] = new_features2['population'] / 5
    # new_features2['year'] = new_features2['year'] + 1000
    # new_features2['temperature'] = new_features2['temperature'] - 50
    # new_features2['precipitation'] = new_features2['precipitation'] + 1

    # mse2, prediction2, new2 = m.prediction(data2, min_error2, new_prediction=True, new_features=new_features2)
    # print("Mean Squared Error:", mse2)
    # m.plot_prediction(past2, "Anopheles quadrimaculatus occurrence in California in 2022")
    # m.plot_prediction(prediction2, "Anopheles quadrimaculatus occurrence in California by prediction on test features")
    # m.plot_prediction(new2, "Anopheles quadrimaculatus occurrence in California in a given condition")

    # Culex tarsalis
    mosquito3_2022 = mosquito3[mosquito3['year'] == 2022]
    past3 = m.ca_geomosquito(mosquito3_2022)
    data3 = m.merge_all_data(mosquito3)

    min_error3 = m.decide_depth(data3)
    # decide new features
    new_features3 = m.prediction(data3, return_featues=True)
    new_features3['population'] = new_features3['population'] / 5
    new_features3['year'] = new_features3['year'] + 1000
    new_features3['temperature'] = new_features3['temperature'] - 50
    new_features3['precipitation'] = new_features3['precipitation'] + 1

    mse3, prediction3, new3 = m.prediction(data3, min_error3, new_prediction=True, new_features=new_features3)
    print("Mean Squared Error:", mse3)
    m.plot_prediction(past3, "Anopheles quadrimaculatus occurrence in California in 2022")
    m.plot_prediction(prediction3, "Anopheles quadrimaculatus occurrence in California by prediction on test features")
    m.plot_prediction(new3, "Anopheles quadrimaculatus occurrence in California in a given condition")

    # TODO: plot features
    city_df = m.generate_city_df()
    city_df.plot()
    plt.show()
    plt.close()
    pop_df = m.combine_pop_df()
    pop_df.plot(legend='County')
    plt.show()
    plt.close()



if __name__ == '__main__':
    main()
