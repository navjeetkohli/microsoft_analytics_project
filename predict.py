import pickle
from train import train_model
import os

# Loading the pickle files and Calling the predict function


def do_prediction(features):
    files = os.listdir()
    if 'pipe_pkl' not in files:
        train_model()

    with open('pipe_pkl', 'rb') as f:
        pickle_clf = pickle.load(f)

    # Do prediction and evaluting the prediction
    prediction = pickle_clf.predict(features)

    return prediction
