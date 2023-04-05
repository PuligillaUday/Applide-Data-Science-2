import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


def read_data(filename):
    """
    Reads data from a CSV file and returns cleaned dataframes with years and countries as columns.

    Parameters:
    filename (str): Name of the CSV file to read data from.

    Returns:
    df_years (pandas.DataFrame): Dataframe with years as columns and countries and indicators as rows.
    df_countries (pandas.DataFrame): Dataframe with countries as columns and years and indicators as rows.
    """
    # read the CSV file and skip the first 4 rows
    df = pd.read_csv(filename, skiprows=4)

    # drop unnecessary columns
    cols_to_drop = ['Country Code', 'Indicator Code', 'Unnamed: 66']
    df = df.drop(cols_to_drop, axis=1)

    # rename remaining columns
    df = df.rename(columns={'Country Name': 'Country'})

    # melt the dataframe to convert years to a single column
    df = df.melt(id_vars=['Country', 'Indicator Name'],
                 var_name='Year', value_name='Value')

    # convert year column to integer and value column to float
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')

    # separate dataframes with years and countries as columns
    df_years = df.pivot_table(
        index=['Country', 'Indicator Name'], columns='Year', values='Value')
    df_countries = df.pivot_table(
        index=['Year', 'Indicator Name'], columns='Country', values='Value')

    # clean the data
    df_years = df_years.dropna(how='all', axis=1)
    df_countries = df_countries.dropna(how='all', axis=1)

    return df_years, df_countries


def calculate_summary_statsss(df_years):
    # create a dictionary to store the summary statistics
    summary_stats = {}

    # select the indicators and countries of interest
    indicators = ["Mortality rate, under-5 (per 1,000 live births)", "Population in urban agglomerations of more than 1 million (% of total population)"]
    countries = ["United States", "China", "India"]

    # calculate summary statistics for each indicator and country
    for indicator in indicators:
        for country in countries:
            # summary statistics for individual countries
            stats = df_years.loc[(country, indicator)].describe()
            summary_stats[f'{country} - {indicator}'] = stats
            # calculate median and standard deviation
            median = df_years.loc[(country, indicator)].median()
            std = df_years.loc[(country, indicator)].std()
            summary_stats[f'{country} - {indicator} Median'] = median
            summary_stats[f'{country} - {indicator} Standard Deviation'] = std

        # summary statistics for the world
        stats = df_years.loc[('World', indicator)].describe()
        summary_stats[f'World - {indicator}'] = stats
        # calculate median and standard deviation
        median = df_years.loc[('World', indicator)].median()
        std = df_years.loc[('World', indicator)].std()
        summary_stats[f'World - {indicator} Median'] = median
        summary_stats[f'World - {indicator} Standard Deviation'] = std

    return summary_stats


def print_summary_stats(summary_stats):
    """
    Prints the summary statistics.

    Args:
    - summary_stats (dict): Dictionary containing the summary statistics.
    """
    # print the summary statistics
    for key, value in summary_stats.items():
        print(key)
        print(value)
        print()


def create_scatter_plots(df_years, indicators, countries):
    """
    Creates scatter plots for the given indicators and countries.

    Args:
    - df_years (pandas.DataFrame): DataFrame containing the data for each country and indicator across years.
    - indicators (list of str): List of indicators to create scatter plots for.
    - countries (list of str): List of countries to create scatter plots for.
    """
    for country in countries:
        for i in range(len(indicators)):
            for j in range(i+1, len(indicators)):
                x = df_years.loc[(country, indicators[i])]
                y = df_years.loc[(country, indicators[j])]
                plt.scatter(x, y)
                plt.xlabel(indicators[i])
                plt.ylabel(indicators[j])
                plt.title(country)
                plt.show()

def subset_data(df_years, countries, indicators):
    """
    Subsets the data to include only the selected countries and indicators.
    Returns the subsetted data as a new DataFrame.
    """
    df = df_years.loc[(countries, indicators), :]
    df = df.transpose()
    return df

def calculate_correlations(df):
    """
    Calculates the correlations between the indicators in the input DataFrame.
    Returns the correlation matrix as a new DataFrame.
    """
    corr = df.corr()
    return corr

def visualize_correlations(corr):
    """
    Plots the correlation matrix as a heatmap using Seaborn.
    """
    sns.heatmap(corr, cmap='coolwarm', annot=True, square=True)
    plt.title('Correlation Matrix of Indicators')
    plt.show()

def plot_foreign_direct_investment(df_years):
    country_list = ['United States', 'India', 'China', 'Japan']
    indicator = 'Foreign direct investment, net inflows (% of GDP)'
    for country in country_list:
        df_subset = df_years.loc[(country, indicator), :]
        plt.plot(df_subset.index, df_subset.values, label=country)
    plt.xlabel('Year')
    plt.ylabel(indicator)
    plt.title('Foreign direct investment, net inflows (% of GDP)')
    plt.legend()
    plt.show()


def plot_population(df_years):
  country_list = ['United States', 'India', 'China']
  energy_indicator = 'Population, total'
  emissions_indicator = 'Urban population'
  years = [1960, 1970, 1980, 1990, 2000]
  x = np.arange(len(country_list))
  width = 0.35

  fig, ax = plt.subplots()

  for i, year in enumerate(years):
      energy_values = []
      emissions_values = []
      for country in country_list:
          energy_values.append(
              df_years.loc[(country, energy_indicator), year])
          emissions_values.append(
              df_years.loc[(country, emissions_indicator), year])
      rects1 = ax.bar(x - width/2 + i*width/len(years), energy_values,
                      width/len(years), label=str(year)+" "+energy_indicator)
      rects2 = ax.bar(x + width/2 + i*width/len(years), emissions_values,
                      width/len(years), label=str(year)+" "+emissions_indicator)

  ax.set_xlabel('Country')
  ax.set_ylabel('Value')
  ax.set_title('Total Population Vs Urban Population')
  ax.set_xticks(x)
  ax.set_xticklabels(country_list)
  ax.legend(loc='upper left', fontsize=10, title='Legend Title')
  fig.tight_layout()
  plt.show()


if __name__ == '__main__':
    df_years, df_countries = read_data(
        r"C:\Users\navee\Downloads\API_19_DS2_en_csv_v2_5361599.csv")

    # Call plot_gdp_per_capita to create the GDP per capita plot
    plot_foreign_direct_investment(df_years)

    # Call plot_energy_use_and_co2_emissions to create the energy use and CO2 emissions plot
    plot_population(df_years)

   # select the indicators of interest
indicators = ['Population, total', 'CO2 emissions (kt)']

# select a few countries for analysis
countries = ['United States', 'China', 'India', 'Japan']

# calculate summary statistics
#summary_stats = calculate_summary_stats(df_years, countries, indicators)

# print the summary statistics
#print_summary_stats(summary_stats)


summary_statsss = calculate_summary_statsss(df_years)
print(summary_statsss)

# Use the describe method to explore the data for the 'United States'
us_data = df_years.loc[('United States', slice(None)), :]
us_data_describe = us_data.describe()
print("Data for United States")
print(us_data_describe)

# Use the mean method to find the mean GDP per capita for each country
gdp_per_capita = df_years.loc[(slice(None), 'Population, total'), :]
gdp_per_capita_mean = gdp_per_capita.mean()
print("\nMean GDP per capita for each country")
print(gdp_per_capita_mean)

# Use the sum method to find the total CO2 emissions for each year
co2_emissions = df_years.loc[(
    slice(None), 'CO2 emissions (metric tons per capita)'), :]
co2_emissions_sum = co2_emissions.sum()
print("\nTotal CO2 emissions for each year")
print(co2_emissions_sum)

df = subset_data(df_years, countries, indicators)
corr = calculate_correlations(df)
#showing heat map
visualize_correlations(corr)

# create scatter plots
create_scatter_plots(df_years, indicators, countries)
