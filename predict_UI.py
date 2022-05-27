from imports import *
from clean import *
from predict import *

# UI for the prediction page


def predict_data():

    st.subheader('Fill the Data to Predict Selling Price for your Car')

    year, km = st.columns(2)

    year_buy = year.number_input(
        'Year you bought the car in', step=1., format='%.0f')
    kilo = km.number_input('Km Driven', step=1., format='%.1f')

    seller_type, fuel_type, trans_type = st.columns(3)

    seller = seller_type.selectbox(
        'Seller Type', ['None', 'Dealer', 'Individual', 'Trustmark Dealer'], index=0)
    fuel = fuel_type.selectbox(
        'Fuel Type', ['None', 'Diesel', 'Petrol', 'CNG', 'LPG', 'Electric'], index=0)
    trans = trans_type.selectbox(
        'Transmission Type', ['None', 'Manual', 'Automatic'], index=0)

    mileage, seats = st.columns(2)

    mil = mileage.number_input('Enter Mileage of your Car', step=0.1)
    seat = seats.slider('Enter Number of Seats in your Car',
                        0.0, 10.0, step=1., format='%.1f')

    company, max_power, engine = st.columns(3)

    com = company.selectbox('Company', ['None', 'Maruti', 'Hyundai', 'Ford', 'Mahindra', 'Tata', 'Renault',
                                        'Nissan', 'Mercedes-Benz', 'Toyota', 'Volkswagen',
                                        'Honda', 'Chevrolet', 'BMW', 'Skoda', 'others'], index=0)
    max = max_power.number_input('Enter Max Power of your Car (BHP)', step=0.1)
    eng = engine.number_input(
        'Enter Engine cc of your Car', step=1., format='%.1f')

    a, but, c = st.columns(3)
    clicked = but.button('Predict Selling Price')

    if clicked:
        if seller == 'None':
            st.warning('No Null values Allowed! Please fill all the details.')
        elif fuel == 'None':
            st.warning('No Null values Allowed! Please fill all the details.')
        elif trans == 'None':
            st.warning('No Null values Allowed! Please fill all the details.')
        elif com == 'None':
            st.warning('No Null values Allowed! Please fill all the details.')
        elif year_buy == 0.0:
            st.warning('No Null values Allowed! Please fill all the details.')
        elif kilo == 0.0:
            st.warning('No Null values Allowed! Please fill all the details.')
        elif seat == 0.0:
            st.warning('No Null values Allowed! Please fill all the details.')
        elif eng == 0.0:
            st.warning('No Null values Allowed! Please fill all the details.')
        elif mil == 0.00:
            st.warning('No Null values Allowed! Please fill all the details.')
        elif max == 0.00:
            st.warning('No Null values Allowed! Please fill all the details.')
        else:
            features = np.array(
                [[year_buy, seller, kilo, fuel, trans, mil, eng, max, seat, com]])
            pred = do_prediction(features)
            st.success('Your Car\'s Predicted Selling Price is Rs. {}'.format(
                np.round(pred, 2)[0]))
