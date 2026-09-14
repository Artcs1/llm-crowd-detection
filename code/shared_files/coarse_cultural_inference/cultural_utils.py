import numpy as np
import pandas as pd

country_to_region_wvs = {
    "United Kingdom": "English-speaking",
    "Uruguay": "Latin America",
    "Belgium": "Catholic Europe",
    "United States": "English-speaking",
    "Kenya": "African-Islamic",
    "Japan": "Confucian Asia",
    "Jordan": "African-Islamic",
    "Canada": "English-speaking",
    "Peru": "Latin America",
    "United Arab Emirates": "African-Islamic",
    "Azerbaijan": "African-Islamic",
    "Thailand": "West & South Asia",
    "Syria": "African-Islamic",
    "Ireland": "English-speaking",
    "Chile": "Latin America",
    "Malawi": "African-Islamic",
    "Poland": "Catholic Europe",
    "South Korea": "Confucian Asia",
    "Estonia": "Protestant Europe",
    "Jamaica": "English-speaking",
    "Brazil": "Latin America",
    "Iraq": "African-Islamic",
    "Germany": "Protestant Europe",
    "Colombia": "Latin America",
    "Norway": "Protestant Europe",
    "Holy See": "Catholic Europe",
    "Mexico": "Latin America",
    "Italy": "Catholic Europe",
    "Taiwan": "Confucian Asia",
    "China": "Confucian Asia",
    "Finland": "Protestant Europe",
    "France": "Catholic Europe",
    "Bangladesh": "African-Islamic",
    "Australia": "English-speaking",
    "Singapore": "West & South Asia",
    "Georgia": "Orthodox Europe",
    "Portugal": "Catholic Europe",
    "Gambia": "African-Islamic",
    "Belarus": "Orthodox Europe",
    "Egypt": "African-Islamic",
    "South Africa": "African-Islamic",
    "Kazakhstan": "African-Islamic",
    "Sri Lanka": "West & South Asia",
    "New Zealand": "English-speaking",
    "Vietnam": "West & South Asia",
    "Malta": "Catholic Europe",
    "Eritrea": "African-Islamic",
    "Nepal": "West & South Asia",
    "Ethiopia": "African-Islamic",
    "Antigua and Barbuda": "Latin America",
    "Switzerland": "Protestant Europe",
    "Lebanon": "African-Islamic",
    "Afghanistan": "African-Islamic",
    "Latvia": "Catholic Europe",
    "Costa Rica": "Latin America",
    "Paraguay": "Latin America",
    "Pakistan": "African-Islamic",
    "Indonesia": "West & South Asia",
    "Spain": "Catholic Europe",
    "Morocco": "African-Islamic",
    "Serbia": "Orthodox Europe",
    "Czech Republic": "Catholic Europe",
    "Nigeria": "African-Islamic",
    "Hungary": "Catholic Europe",
    "Greece": "Orthodox Europe",
    "Bermuda": "English-speaking",
    "Moldova": "Orthodox Europe",
    "Iceland": "Protestant Europe",
    "Denmark": "Protestant Europe",
    "Slovakia": "Catholic Europe",
    "Saint Lucia": "Latin America",
    "Sweden": "Protestant Europe",
    "Mali": "African-Islamic",
    "Venezuela": "Latin America",
    "Tajikistan": "African-Islamic",
    "Israel": "West & South Asia",
    "Kyrgyzstan": "African-Islamic",
    "Barbados": "English-speaking",
    "Kuwait": "African-Islamic",
    "Benin": "African-Islamic",
    "Madagascar": "African-Islamic",
    "India": "West & South Asia",
    "Ghana": "African-Islamic",
    "Samoa": "Other",
    "Austria": "Catholic Europe",
    "Brunei": "West & South Asia",
    "Romania": "Orthodox Europe",
    "Senegal": "African-Islamic",
    "Turkey": "African-Islamic",
    "Netherlands": "Protestant Europe",
}


country_to_region_globe = {
    "United Kingdom": "Anglo",
    "Belgium": "Germanic Europe",
    "Ireland": "Anglo",
    "France": "Latin Europe",
    "Germany": "Germanic Europe",
    "Portugal": "Latin Europe",
    "Spain": "Latin Europe",
    "Italy": "Latin Europe",
    "Switzerland": "Latin Europe",
    "Netherlands": "Germanic Europe",
    "Austria": "Germanic Europe",
    "Malta": "Latin Europe",
    "Denmark": "Nordic Europe",
    "Norway": "Nordic Europe",
    "Sweden": "Nordic Europe",
    "Finland": "Nordic Europe",
    "Iceland": "Nordic Europe",
    "Greece": "Eastern Europe",
    "Poland": "Eastern Europe",
    "Czech Republic": "Eastern Europe",
    "Slovakia": "Eastern Europe",
    "Hungary": "Eastern Europe",
    "Romania": "Latin Europe",
    "Belarus": "Eastern Europe",
    "Serbia": "Eastern Europe",
    "Moldova": "Latin Europe",
    "Estonia": "Nordic Europe",
    "Latvia": "Nordic Europe",
    "United States": "Anglo",
    "Canada": "Anglo",
    "Bermuda": "Other",
    "Mexico": "Latin America",
    "Brazil": "Latin America",
    "Chile": "Latin America",
    "Peru": "Latin America",
    "Uruguay": "Latin America",
    "Paraguay": "Latin America",
    "Colombia": "Latin America",
    "Costa Rica": "Latin America",
    "Venezuela": "Latin America",
    "Jamaica": "African",
    "Barbados": "African",
    "Antigua and Barbuda": "African",
    "Saint Lucia": "Other",
    "Kenya": "African",
    "Malawi": "African",
    "South African": "African",
    "Nigeria": "African",
    "Ghana": "African",
    "Senegal": "African",
    "Benin": "African",
    "Madagascar": "African",
    "Ethiopia": "African",
    "Eritrea": "African",
    "Mali": "African",
    "Gambia": "African",
    "United Arab Emirates": "Middle East",
    "Jordan": "Middle East",
    "Syria": "Middle East",
    "Iraq": "Middle East",
    "Lebanon": "Middle East",
    "Egypt": "Middle East",
    "Morocco": "Middle East",
    "Kuwait": "Middle East",
    "Israel": "Latin Europe",
    "Turkey": "Middle East",
    "Afghanistan": "South-East Asia",
    "India": "South-East Asia",
    "Pakistan": "South-East Asia",
    "Bangladesh": "South-East Asia",
    "Sri Lanka": "South-East Asia",
    "Nepal": "South-East Asia",
    "China": "Confucian Asia",
    "Japan": "Confucian Asia",
    "South Korea": "Confucian Asia",
    "Taiwan": "Confucian Asia",
    "Thailand": "South-East Asia",
    "Vietnam": "Confucian Asia",
    "Indonesia": "South-East Asia",
    "Singapore": "Confucian Asia",
    "Brunei": "South-East Asia",
    "Kazakhstan": "Eastern Europe",
    "Kyrgyzstan": "Eastern Europe",
    "Tajikistan": "South-East Asia",
    "Azerbaijan": "Middle East",
    "Georgia": "Eastern Europe",
    "Australia": "Anglo",
    "New Zealand": "Anglo",
    "Samoa": "South-East Asia",
    "Holy See": "Other"
}

country_to_iso3 = {
    'United Kingdom': 'GBR',
    'Uruguay': 'URY',
    'Belgium': 'BEL',
    'United States': 'USA',
    'Kenya': 'KEN',
    'Japan': 'JPN',
    'Jordan': 'JOR',
    'Canada': 'CAN',
    'Peru': 'PER',
    'United Arab Emirates': 'ARE',
    'Azerbaijan': 'AZE',
    'Thailand': 'THA',
    'Syria': 'SYR',
    'Ireland': 'IRL',
    'Chile': 'CHL',
    'Malawi': 'MWI',
    'Poland': 'POL',
    'South Korea': 'KOR',
    'Estonia': 'EST',
    'Jamaica': 'JAM',
    'Brazil': 'BRA',
    'Iraq': 'IRQ',
    'Germany': 'DEU',
    'Colombia': 'COL',
    'Norway': 'NOR',
    'Holy See': 'VAT',
    'Mexico': 'MEX',
    'Italy': 'ITA',
    'Taiwan': 'TWN',
    'China': 'CHN',
    'Finland': 'FIN',
    'France': 'FRA',
    'Bangladesh': 'BGD',
    'Australia': 'AUS',
    'Singapore': 'SGP',
    'Georgia': 'GEO',
    'Portugal': 'PRT',
    'Gambia': 'GMB',
    'Belarus': 'BLR',
    'Egypt': 'EGY',
    'South Africa': 'ZAF',
    'Kazakhstan': 'KAZ',
    'Sri Lanka': 'LKA',
    'New Zealand': 'NZL',
    'Vietnam': 'VNM',
    'Malta': 'MLT',
    'Eritrea': 'ERI',
    'Nepal': 'NPL',
    'Ethiopia': 'ETH',
    'Antigua and Barbuda': 'ATG',
    'Switzerland': 'CHE',
    'Lebanon': 'LBN',
    'Afghanistan': 'AFG',
    'Latvia': 'LVA',
    'Costa Rica': 'CRI',
    'Paraguay': 'PRY',
    'Pakistan': 'PAK',
    'Indonesia': 'IDN',
    'Spain': 'ESP',
    'Morocco': 'MAR',
    'Serbia': 'SRB',
    'Czech Republic': 'CZE',
    'Nigeria': 'NGA',
    'Hungary': 'HUN',
    'Greece': 'GRC',
    'Bermuda': 'BMU',
    'Moldova': 'MDA',
    'Iceland': 'ISL',
    'Denmark': 'DNK',
    'Slovakia': 'SVK',
    'Saint Lucia': 'LCA',
    'Sweden': 'SWE',
    'Mali': 'MLI',
    'Venezuela': 'VEN',
    'Tajikistan': 'TJK',
    'Israel': 'ISR',
    'Kyrgyzstan': 'KGZ',
    'Barbados': 'BRB',
    'Kuwait': 'KWT',
    'Benin': 'BEN',
    'Madagascar': 'MDG',
    'India': 'IND',
    'Ghana': 'GHA',
    'Samoa': 'WSM',
    'Austria': 'AUT',
    'Brunei': 'BRN',
    'Romania': 'ROU',
    'Senegal': 'SEN',
    'Turkey': 'TUR',
    'Netherlands': 'NLD'
}



def merge_list(x):
    l = []
    for sublist in x:
        l.extend(sublist)
    return l



def uniform_sample_region(g, N, seed):
    countries = g['country'].unique()
    n_countries = len(countries)
    
    # how many videos to sample per country
    n_per_country = max(1, N // n_countries)

    samples = []
    for country, g_country in g.groupby('country'):
        # sample videos for this country
        n_videos = min(n_per_country, g_country['video'].nunique())
        sampled_videos = (
            g_country['video'].drop_duplicates()
            .sample(n=n_videos, random_state=seed)
        )
        # sample one row per video
        rows = (
            g_country[g_country['video'].isin(sampled_videos)]
            .groupby('video', group_keys=False)[g_country.columns]
            .apply(lambda v: v.sample(n=1, random_state=seed))
        )
        samples.append(rows)
    
    df_region = pd.concat(samples)
    
    # if we have fewer than N (because some countries lacked enough videos), top up
    if len(df_region['video'].unique()) < N:
        remaining = N - len(df_region['video'].unique())
        # sample remaining videos uniformly from region
        extra_videos = (
            g[~g['video'].isin(df_region['video'])]
            ['video'].drop_duplicates()
            .sample(
                n=min(remaining, g['video'].nunique() - len(df_region['video'].unique())),
                random_state=seed
            )
        )
        extra_rows = (
            g[g['video'].isin(extra_videos)]
            .groupby('video', group_keys=False)[g.columns]
            .apply(lambda v: v.sample(n=1, random_state=seed))
        )
        df_region = pd.concat([df_region, extra_rows])
    
    return df_region


def get_final_df(results, closeness_df):
    final_df = pd.DataFrame(closeness_df)
    final_df['Hand Holding'] = 0
    final_df['Hugging'] = 0

    for r in results:
        final_df['Hand Holding'] += r['Hand Holding']
        final_df['Hugging'] += r['Hugging']

    final_df['Hand Holding'] /= len(results)
    final_df['Hugging'] /= len(results)

    # final_df['Hand Holding Norm'] = final_df['Hand Holding'] / final_df['Row Counts']
    # final_df['Hugging Norm'] = final_df['Hugging'] / final_df['Row Counts']

    final_df['Total'] = (final_df['Hand Holding'] + final_df['Hugging'])/2
    # final_df['Total Norm'] = final_df['Hand Holding Norm'] + final_df['Hugging Norm']
    final_df.sort_values('Total', ascending=False, inplace=True)

    return final_df