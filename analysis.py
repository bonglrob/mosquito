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


def main():
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

    btn_04_33 = (mosquito1['year'] >= 1904) & (mosquito1['year'] <= 1933)
    btn_34_63 = (mosquito1['year'] >= 1934) & (mosquito1['year'] <= 1963)
    btn_64_93 = (mosquito1['year'] >= 1964) & (mosquito1['year'] <= 1993)
    btn_94_23 = (mosquito1['year'] >= 1994) & (mosquito1['year'] <= 2023) &\
                (mosquito1['stateProvince'] != 'Hawaii')

    occurrence_04_33 = mosquito1[btn_04_33]
    occurrence_34_63 = mosquito1[btn_34_63]
    occurrence_64_93 = mosquito1[btn_64_93]
    occurrence_94_23 = mosquito1[btn_94_23]

    occurrence_04_33 =\
        m.filter_occurrence_by_30_year(occurrence_04_33, '1')
    occurrence_34_63 =\
        m.filter_occurrence_by_30_year(occurrence_34_63, '2')
    occurrence_64_93 =\
        m.filter_occurrence_by_30_year(occurrence_64_93, '3')
    occurrence_94_23 =\
        m.filter_occurrence_by_30_year(occurrence_94_23, '4')

    fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, figsize=(20, 10))
    us_map.plot(color='#EEEEEE', edgecolor='#FFFFFF', ax=ax1)
    occurrence_points = \
        gpd.GeoDataFrame(occurrence_04_33, geometry='coordinates1')
    occurrence_points.plot(column='coordinates1', markersize=5, ax=ax1,
                           vmin=0, vmax=1)
    ax1.set_title('Occurrences of Aedes aegypti in 1903-1933')

    us_map.plot(color='#EEEEEE', edgecolor='#FFFFFF', ax=ax2)
    occurrence_points = gpd.GeoDataFrame(occurrence_34_63,
                                         geometry='coordinates2')
    occurrence_points.plot(column='coordinates2', markersize=5, ax=ax2,
                           vmin=0, vmax=1)
    ax2.set_title('Occurrences of Aedes aegypti in 1934-1963')

    us_map.plot(color='#EEEEEE', edgecolor='#FFFFFF', ax=ax3)
    occurrence_points = gpd.GeoDataFrame(occurrence_64_93,
                                         geometry='coordinates3')
    occurrence_points.plot(column='coordinates3', markersize=5, ax=ax3,
                           vmin=0, vmax=1)
    ax3.set_title('Occurrences of Aedes aegypti in 1964-1993')

    us_map.plot(color='#EEEEEE', edgecolor='#FFFFFF', ax=ax4)
    occurrence_points = gpd.GeoDataFrame(occurrence_94_23,
                                         geometry='coordinates4')
    occurrence_points.plot(column='coordinates4', markersize=5, ax=ax4,
                           vmin=0, vmax=1)
    ax4.set_title('Occurrences of Aedes aegypti in 1994-2023')
    plt.savefig('./results/Occurrence_Aedes_aegypti.png')

    plt.show()
    plt.close()

    # question 2 :
    # Which species occurs more in the United States?

    # Filter Aedes species by month from 2002 - 2022
    aedes_species_count = m.get_count_per_month(mosquito1)
    aedes_species_count = m.filter_years(aedes_species_count)
    aedes_species_count = m.add_month_name(aedes_species_count)

    # Filter Anopheles species by month from 2002- 2022
    anopheles_species_count = m.get_count_per_month(mosquito2)
    anopheles_species_count = m.filter_years(anopheles_species_count)
    anopheles_species_count = m.add_month_name(anopheles_species_count)

    # Filter Culex species by month from 2002 - 2022
    culex_species_count = m.get_count_per_month(mosquito3)
    culex_species_count = m.filter_years(culex_species_count)
    culex_species_count = m.add_month_name(culex_species_count)

    # Merge all filtered species dataset
    all_species_count = m.merge_all_species_data(aedes_species_count,
                                                 anopheles_species_count,
                                                 culex_species_count)

    # Plot both all species and aedes species count by month
    m.plot_species(all_species_count)
    m.plot_aedes(aedes_species_count)

    # question 3: Mosquito occurrence prediction
    # Aedes aegypti
    print("Aedes aegypti")
    print()

    # read DataFrame to GeoDataFrame
    past1 = m.ca_geomosquito(mosquito1)
    data1 = m.merge_all_data(mosquito1)
    print(data1.head())
    data1.to_csv('./results/Aedes aegypti data.csv', index=False)
    print()

    # Decide the depth of trees
    min_error1 = m.decide_depth(data1, "Aedes aegypti")

    # TODO: edit new features if you want
    new_features1 = m.prediction(data1, min_error1, return_features=True)
    new_features1['population'] = new_features1['population'] + 10000
    new_features1['year'] = new_features1['year'].apply(int) + 50
    new_features1['temperature'] = new_features1['temperature'] + 5
    new_features1['precipitation'] = new_features1['precipitation'] + 1

    # plot dataset and prediction
    mse1, prediction1, new1 = \
        m.prediction(data1, min_error1, new_prediction=True,
                     new_features=new_features1)
    print("Mean Squared Error:", mse1)
    print()
    m.plot_prediction(past1, "Aedes aegypti occurrence in California in total")
    title1 = "Aedes aegypti occurrence in California by prediction " + \
        "on test datasets"
    m.plot_prediction(prediction1, title1)
    title2 = "Aedes aegypti occurrence prediction in California"
    m.plot_prediction(new1, title2)
    print()

    # Anopheles quadrimaculatus
    print("Anopheles quadrimaculatus")
    print()

    # read DataFrame to GeoDataFrame
    m.ca_geomosquito(mosquito2)
    print()

    # Culex tarsalis
    print("Culex tarsalis")
    print()

    # read DataFrame to GeoDataFrame
    past3 = m.ca_geomosquito(mosquito3)
    data3 = m.merge_all_data(mosquito3)
    print(data3.head())
    data3.to_csv('./results/Culex tarsalis data.csv', index=False)
    print()

    # Decide the depth of trees
    min_error3 = m.decide_depth(data3, "Culex tarsalis", random=999)

    # TODO: edit new features if you want
    new_features3 = m.prediction(data3, min_error1, return_features=True)
    new_features3['population'] = new_features3['population'] + 10000
    new_features3['year'] = new_features3['year'].apply(int) + 50
    new_features3['temperature'] = new_features3['temperature'] + 5
    new_features3['precipitation'] = new_features3['precipitation'] + 1

    # plot dataset and prediction
    mse3, prediction3, new3 = \
        m.prediction(data3, min_error3, new_prediction=True,
                     new_features=new_features3, random=999)
    print("Mean Squared Error:", mse3)
    print()
    m.plot_prediction(past3,
                      "Culex tarsalis occurrence in California in total")
    title3 = "Culex tarsalis occurrence in California by prediction" + \
        "on test datasets"
    m.plot_prediction(prediction3, title3)
    title4 = "Culex tarsalis occurrence prediction in California"
    m.plot_prediction(new3, title4)


if __name__ == '__main__':
    main()
