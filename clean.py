from imports import *

import warnings
warnings.filterwarnings('ignore')

#Step 1 of cleaning data for the Graphs
def clean_data():
    cars_data = pd.read_csv('Cardekho_Extract.csv')

    cars_data.drop(['Source.Name', 'web-scraper-order', 'web-scraper-start-url'], axis = 1, inplace=True)

    cars_data.dropna(axis = 0, how = 'all', inplace= True)

    cars_data['selling_price'] = cars_data['selling_price'].apply(str).str.replace('*','')
    cars_data['selling_price'] = cars_data['selling_price'].apply(str).str.replace(',','')
    cars_data[['selling_price','unit']] = cars_data['selling_price'].str.split(n=1, expand=True)
    cars_data['selling_price'] = cars_data['selling_price'].astype('float64', errors= 'raise')

    cars_data.loc[cars_data.unit == "Lakh", 'selling_price'] = cars_data['selling_price']*100000.0
    cars_data.loc[cars_data.unit == "Cr", 'selling_price'] = cars_data['selling_price']*10000000.0

    cars_data = cars_data.drop(['unit'], axis =1)

    cars_data['mileage'] = cars_data['mileage'].str.split(' ', expand =True)[0].str.split('e',n=2, expand=True)[2]
    cars_data['engine'] = cars_data['engine'].str.split(' ', expand=True)[0].str.split('e', expand=True)[1]
    cars_data['max_power'] = cars_data['max_power'].str.split(' ', expand=True)[1].str.split('r', expand=True)[1]
    cars_data['seats'] = cars_data['seats'].str.split('s', expand=True)[1]
    cars_data['km_driven'] = cars_data['km_driven'].str.split(' ', n=1, expand=True)[0]
    cars_data['km_driven'] = cars_data['km_driven'].str.replace(',','')

    rep_cols = ["engine", "max_power", "seats"]
    cars_data[rep_cols] = cars_data[rep_cols].replace(r'[^\d.]+' , '', regex=True)
    cars_data[rep_cols] = cars_data[rep_cols].replace('', '0', regex=True)

    cars_data= cars_data.astype({'km_driven': 'float64', 'mileage': 'float64', 'engine': 'float64', 'max_power': 'float64', 'seats': 'float64'})

    cars_data['company'] = cars_data['full_name'].str.split(' ', expand=True)[0]

    cars_data.drop(['new-price', 'owner_type', 'full_name'], axis = 1, inplace=True)

    return cars_data