import streamlit as st
import pandas as pd
import numpy as np
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import pickle
from sklearn.ensemble import RandomForestRegressor


def Preprocess_1( X):
    X_copy = X.copy()  # Create a copy of X to avoid modifying the original

    X_copy['aromaticity'] = X_copy['protein_sequence'].apply(lambda sequence: ProteinAnalysis(sequence).aromaticity())
    X_copy['molecular_weight'] = X_copy['protein_sequence'].apply(lambda sequence: ProteinAnalysis(sequence).molecular_weight())
    X_copy['instability_index'] = X_copy['protein_sequence'].apply(lambda sequence: ProteinAnalysis(sequence).instability_index())
    X_copy['hydrophobicity'] = X_copy['protein_sequence'].apply(lambda sequence: ProteinAnalysis(sequence).gravy(scale='KyteDoolitle'))
    X_copy['isoelectric_point'] = X_copy['protein_sequence'].apply(lambda sequence: ProteinAnalysis(sequence).isoelectric_point())
    X_copy['charge_at_pH'] = X_copy.apply(lambda row: ProteinAnalysis(row['protein_sequence']).charge_at_pH(row['pH']), axis=1)

    X_copy['protien_length'] = X_copy['protein_sequence'].apply(len)
    amino_acids = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']
    for i in amino_acids:
        X_copy[i] = X_copy['protein_sequence'].str.count(i) / X_copy['protien_length']
    X_copy.drop(columns=['protein_sequence', 'protien_length'], inplace=True)

    return X_copy



st.title('Web Deployment of Enzyme Thermostability  App')
st.subheader('To Predict the Thermostability at given pH based on protien structure')
st.set_option('deprecation.showPyplotGlobalUse', False)
with open('random_forest_model.pkl', 'rb') as file:
    model_reg = pickle.load(file)    
file.close()


protein_sequence = st.text_input('Enter the Protien Sequence','AAADGEPLHNEEERAGAGQVGRSLPQESEEQRTGSRPRRRRDLGSRLQAQRRAQRVAWEDGDENVGQTVIPAQEEEGIEKPAEVHPTGKIGAKKLRKLEEKQARKAQREAEEAEREERKRLESQREAEWKKEEERLRLKEEQKEEEERKAQEEQARREHEEYLKLKEAFVVEEEGVSETMTEEQSHSFLTEFINYIKKSKVVLLEDLAFQMGLRTQDAINRIQDLLTEGTLTGVIDDRGKFIYITPEELAAVANFIRQRGRVSITELAQASNSLISWGQDLPAQAS')
pH = st.number_input('pH',1.0,12.0,7.0)
X = pd.DataFrame({'protein_sequence':protein_sequence,'pH':pH},index=[0])
X = Preprocess_1(X)

if st.button('Predict'):
    pred = model_reg.predict(X)
    st.subheader(f'The thermostability of the Enzyme at pH : {np.round(pH,2)} is : {np.round(pred,2)}')

    
    
st.subheader('Made by :')

contact_info = """ https://www.linkedin.com/in/ajish-kurian-daniel/ """

st.markdown(contact_info)
