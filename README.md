# Mosquito Occurence Prediction due to Climate Change

Research Results can be read here [Report](report.pdf)

# How to run Mosquitoes Program
### Open with Visual Studio Code
### Installs
#### In terminal:
pip install shapely <br />
pip install geopandas <br />
pip install plotly==5.13.1 <br />
##### If you use VSCode:
Install Python extension (the one published by Mircosoft)
##### Notes for windows user:
If the above commands report errors, please switch to PyCharm
and type in the commands again.
### Run analysis.py

### How to View Plotly Occurence Line Plots (for research question 2)
Your default browser will open up 2 pages, each a different plot:
1. Monthly Count of Mosquitoes in the US by Species
  - Use the bottom slider to select a year
  - Top right button "Autoscale" will scale the y-axis to the max value of that year
  - Select any species in the legend key to toggle them on and off
2. Monthly Count of Aedes Aegypti from 2002 to 2022
  - Select any year in the legend key to toggle them on and off to focus on years of interest

You can hover any datapoint to view exact number of occurences for that month

### Other files
* mosquito.py: This file implements functions used in analysis.py.
* test.py: This file implements test functions for mosquito.py.