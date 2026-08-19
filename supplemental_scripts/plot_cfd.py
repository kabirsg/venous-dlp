'''
This script just plots Gurnish's CFD data that has already been extracted into a pickle file and plot only that
'''

import pickle
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

def get_cfd_data(case_name):
    

    with open("../Gurnish_Data/Gurnish_Case_Data.pkl", "rb") as f:
        zones_df = pickle.load(f)

    x = zones_df[case_name]["dist"]
    pcen = zones_df[case_name]["pcen"]
    return x, pcen

def plot(x, p, save):
    fig, ax = plt.subplots(1, 1, figsize=(10,6))
    ax.plot(x, p, color="black", linewidth=2, linestyle="--", label="3D CFD")
    ax.set_xlabel("Length Along Centerline [cm]")
    ax.set_ylabel("Pressure Drop [mmHg]")
    ax.set_title("Pressure Drop Along Centerline from 3D CFD")
    ax.legend()

    Path(save).parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(save, dpi=300)
    plt.show()
    

def main():
    CASE_NAME = "Case A individual"
    SAVE_LOC = ""

    x, p = get_cfd_data(CASE_NAME)
    plot(x, p, SAVE_LOC)

if __name__ == "__main__":
    main()