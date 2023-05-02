import pickle
import numpy as np
import pandas as pd

with open('disease_pred.pkl', 'rb') as f:
    model = pickle.load(f)

df = pd.read_csv('Training.csv')
df.drop('Unnamed: 133', axis=1, inplace=True)

def get_symptoms():
    symptom_str = input("Enter symptoms separated by commas: ")
    symptoms = [symptom.strip() for symptom in symptom_str.split(",")]
    return symptoms

user_symptom = get_symptoms()

def map_symptoms(symptoms):
    dataset_symptoms = df.columns.tolist()
    dataset_symptoms.pop()
    mapped_symptoms = [0] * len(dataset_symptoms)

    for symptom in symptoms:
        symptom = symptom.strip().replace(' ', '_')
        if symptom in dataset_symptoms:
            mapped_symptoms[dataset_symptoms.index(symptom)] = 1

    return mapped_symptoms

s = map_symptoms(user_symptom)
s = np.array(s).reshape(1, -1)

prediction = model.predict(s)

print("You May Have "+prediction[0]+".")
print("  Consult a Physician as soon as possible.")
