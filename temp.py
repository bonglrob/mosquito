import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt


def q1():
    data = pd.read_csv('./dataset/Occurence_Aedes_aegypti.csv')
    print(data.columns)
    print(len(data))


if __name__ == '__main__':
    q1()