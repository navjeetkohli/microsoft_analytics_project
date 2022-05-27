from imports import *
from graph import *
from predict_UI import *
from process import *
import json
from streamlit_lottie import st_lottie
from contact_us import *
import warnings
warnings.filterwarnings('ignore')

st.title('PREDICTIFY - Used Car Data Analysis')

#loading files for animations
def load_lottiefiles(filepath: str):
    with open(filepath, 'r') as f:
        return json.load(f)

lottie_graph = load_lottiefiles('animations/anim_graphs.json')
lottie_car = load_lottiefiles('animations/car.json')
lottie_car2 = load_lottiefiles('animations/car2.json')

st.sidebar.title('Navigation Panel')
nav = st.sidebar.radio(' ',['Analyze', 'Prediction', 'Contact Us'])

if nav == 'Analyze':
    st_lottie(
    lottie_graph,
    quality= 'medium',
    width= '40rem',
    height= '20rem',
    )
    graphs()

if nav == 'Prediction':
    st_lottie(
    lottie_car2,
    quality= 'medium',
    width= '40rem',
    height= '15rem',
    )
    predict_data()
    
if nav == 'Contact Us':
    contact()
        
    
