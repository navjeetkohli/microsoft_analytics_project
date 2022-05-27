from imports import *
from clean import *

#Step 2 of cleaning data for training the model and prediction
def process_data():
    cars_data=clean_data()

    cars_data['mileage'] = cars_data['mileage'].replace({'0':np.nan, 0:np.nan})

    #Filling the NULL values in data set
    for i in ['mileage','engine', 'max_power', 'seats']:
        company_name = cars_data[cars_data[i].isnull()]['company'].value_counts().index[0]
        if cars_data[i].nunique()>10:
            values = cars_data[cars_data['company']==company_name][i].mean()
        else:
            values = cars_data[cars_data['company']==company_name][i].median()
            
        cars_data[i].fillna(values, inplace=True)
    
    #Removing Outliers
    cars_data = cars_data[cars_data['selling_price'] < 20000000]
    cars_data = cars_data[cars_data['km_driven'] < 1000000]
    cars_data = cars_data[cars_data['mileage'] < 100]
    cars_data = cars_data[cars_data['engine'] < 6100]
    cars_data = cars_data[cars_data['max_power'] < 530]
    cars_data = cars_data.reset_index(drop=True)

    company_name = cars_data.company.value_counts().index[:15]
    for i in range(cars_data.shape[0]):
        if cars_data['company'][i] in company_name:
            continue
        else:
            cars_data['company'][i] = 'others'
    
    return cars_data