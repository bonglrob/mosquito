"""
CSE 163 Final Project
Kyoko Kurihara, Robert Bonglamphone, Christine Cai

This file is an actual runnable file for our mosquito occurrence analysis.
This code solves three questions:

1. Plot of mosquito occurrence in US
2. Plot of mosquito occurrence changes over years in California
3. Mosquito occurrence prediction in California
    1. The code converts three mosquito occurrence CSV datasets to a
       GeoDataFrame with latitude and longitude information.

    2. In order to write a GeoDataFrame
       for machine learning purposes, the code changes the formats of
       population, temperature, and precipitation files, and combines them all
       with the mosquito occurrence GeoPandaDataFrame.
       - For the population datasets, it writes a new DataFrame by cleaning up
         seven CSV files.
       - For temperature and precipitation datasets, it writes
         a new DataFrame with 12 CSV files.

    3. The machine learning part uses:
       - features: 'population', 'temperature', 'precipitation',
                   'location', etc.
       - lables: mosquito occurrence 'latitude', 'longitude',
                 'individual counts'

       It trains a Random Forest regressor model.

    4. The prediction results are then plotted for each mosquito species.
"""
import mosquito as m
import geopandas as gpd
import matplotlib.pyplot as plt
import plotly.express as px
import calendar
import warnings


def main() -> None:
    warnings.filterwarnings('ignore')

    # read data
    mosquito1 = m.get_df_m(m.get_path('Aedes_aegypti_occurrence.csv'))
    mosquito2 = \
        m.get_df_m(m.get_path('Anopheles_quadrimaculatus_occurrence.csv'))
    mosquito3 = m.get_df_m(m.get_path('Culex_tarsalis_occurrence.csv'))

    # question 1
    # prepare map of US
    us_map = gpd.read_file(m.get_path('gz_2010_us_040_00_5m.json'))
    us_map = us_map[(us_map['NAME'] != 'Alaska') &
                    (us_map['NAME'] != 'Hawaii')]

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
    occurrence_points = \
        gpd.GeoDataFrame(occurrence_04_33, geometry='coordinates1')
    occurrence_points.plot(column='coordinates1', markersize=5, ax=ax1,
                           vmin=0, vmax=1)
    ax1.set_title('Occurrences of yellow fever mosquito in 1903-1933')

    us_map.plot(color='#EEEEEE', edgecolor='#FFFFFF', ax=ax2)
    occurrence_points = gpd.GeoDataFrame(occurrence_34_63,
                                         geometry='coordinates2')
    occurrence_points.plot(column='coordinates2', markersize=5, ax=ax2,
                           vmin=0, vmax=1)
    ax2.set_title('Occurrences of yellow fever mosquito in 1934-1963')

    us_map.plot(color='#EEEEEE', edgecolor='#FFFFFF', ax=ax3)
    occurrence_points = gpd.GeoDataFrame(occurrence_64_93,
                                         geometry='coordinates3')
    occurrence_points.plot(column='coordinates3', markersize=5, ax=ax3,
                           vmin=0, vmax=1)
    ax3.set_title('Occurrences of yellow fever mosquito in 1964-1993')

    us_map.plot(color='#EEEEEE', edgecolor='#FFFFFF', ax=ax4)
    occurrence_points = gpd.GeoDataFrame(occurrence_94_23,
                                         geometry='coordinates4')
    occurrence_points.plot(column='coordinates4', markersize=5, ax=ax4,
                           vmin=0, vmax=1)
    ax4.set_title('Occurrences of yellow fever mosquito in 1994-2023')
    plt.savefig('Occurrence_yellow_fever.png')

    plt.show()
    plt.close()

    # question 2
    aedes_species = mosquito1.copy()
    # species2 = mosquito2.copy()
    culex_species = mosquito3.copy()
    # https://www.wrbu.si.edu/vectorspecies/mosquitoes/tarsalis,
    # Culex tarsalis Coquillett, 1896, usually tracked in

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
    # TODO: edit new features if you want
    year = 2022
    month = 9
    new_features = m.generate_features(year=year, month=month)
    new_features['population'] = new_features['population'] + 10000
    new_features['year'] = new_features['year'].apply(int) + 28
    new_features['temperature'] = new_features['temperature'] + 5
    new_features['precipitation'] = new_features['precipitation'] + 1
    # Aedes aegypti
    print("Aedes aegypti")
    print()
    past1 = m.ca_geomosquito(mosquito1)
    data1 = m.merge_all_data(mosquito1)
    print(data1.head())
    print()

    min_error1 = m.decide_depth(data1)

    mse1, prediction1, new1 = \
        m.prediction(data1, min_error1, new_prediction=True,
                     new_features=new_features)
    print("Mean Squared Error:", mse1)
    print()
    m.plot_prediction(past1, "Aedes aegypti occurrence in California in total")
    m.plot_prediction(prediction1,
                      "Aedes aegypti occurrence in California by prediction on test datasets")
    m.plot_prediction(new1, "Aedes aegypti occurrence in California in September 2050")
    print()

    # Anopheles quadrimaculatus
    print("Anopheles quadrimaculatus")
    print()
    m.ca_geomosquito(mosquito2)
    print()

    # Culex tarsalis
    print("Culex tarsalis")
    print()
    past3 = m.ca_geomosquito(mosquito3)
    data3 = m.merge_all_data(mosquito3)
    print(data3.head())
    print()

    min_error3 = m.decide_depth(data3, 999)

    mse3, prediction3, new3 = \
        m.prediction(data3, min_error3, new_prediction=True,
                     new_features=new_features, random=999)
    print("Mean Squared Error:", mse3)
    print()
    m.plot_prediction(past3,
                      "Culex tarsalis occurrence in California in total")
    m.plot_prediction(prediction3, "Culex tarsalis occurrence in California by prediction on test datasets")
    m.plot_prediction(new3, "Culex tarsalis occurrence in California in September 2050")
    warnings.resetwarnings()


if __name__ == '__main__':
    main()
