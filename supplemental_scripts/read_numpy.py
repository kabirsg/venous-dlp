"""
Reads numpy file and creates pickle file
"""

import numpy as np
import pickle

with np.load('outputs_bernoulli/Ali_default.npz') as data:
    print(f"Array Names: {data.files}")



with open("Gurnish_Data/Gurnish_Case_Data.pkl", "rb") as f:
    df = pickle.load(f)

case_name = "Case A individual"
x = df[case_name]["dist"]
pber = df[case_name]["pber"]
pcen = df[case_name]["pcen"]

print("Done.")