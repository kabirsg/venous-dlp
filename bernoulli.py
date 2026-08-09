'''
Recreating Bernoulli results

This is assuming that we are using Gurnish's centerlines - not sure it would work with other centerlines
'''

import numpy as np
import pandas as pd
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from pathlib import Path
import matplotlib.pyplot as plt
import pickle
import config

PRESET_CASES = {
    'CaseA': {
        'inlet_flow_rate': 3.73,
        'cline': "/home/kabir/masters_files/Gurnish_cases/Good/A/CaseA_centerline_with_metrics.vtp",
        'fig_save_folder': 'outputs_ber',
        'case_name': 'CaseA',
        'gurnish_case_name': 'Case A individual'
    },
    'CaseC': {
        'inlet_flow_rate': 5.00,
        'cline': "/home/kabir/masters_files/Gurnish_cases/Good/C/CaseC_centerline_with_metrics.vtp",
        'fig_save_folder': 'outputs_ber',
        'case_name': 'CaseC',
        'gurnish_case_name': 'Case C individual'
    },
    'CaseE': {
        'inlet_flow_rate': 3.48,
        'cline': "/home/kabir/masters_files/Gurnish_cases/Good/E/CaseE_centerline_with_metrics.vtp",
        'fig_save_folder': 'outputs_ber',
        'case_name': 'CaseE',
        'gurnish_case_name': 'Case E individual'
    },
    'CaseF': {
        'inlet_flow_rate': 7.93,
        'cline': "/home/kabir/masters_files/Gurnish_cases/Good/F/CaseF_centerline_with_metrics.vtp",
        'fig_save_folder': 'outputs_ber',
        'case_name': 'CaseF',
        'gurnish_case_name': 'Case F individual'
    },
    'CaseG': {
        'inlet_flow_rate': 5.18,
        'cline': "/home/kabir/masters_files/Gurnish_cases/Good/G/CaseG_centerline_with_metrics.vtp",
        'fig_save_folder': 'outputs_ber',
        'case_name': 'CaseG',
        'gurnish_case_name': 'Case G individual'
    },
    'CaseH': {
        'inlet_flow_rate': 4.50,
        'cline': "/home/kabir/masters_files/Gurnish_cases/Good/H/CaseH_centerline_with_metrics.vtp",
        'fig_save_folder': 'outputs_ber',
        'case_name': 'CaseH',
        'gurnish_case_name': 'Case H individual'
    },
    'CaseI': {
        'inlet_flow_rate': 5.40,
        'cline': "/home/kabir/masters_files/Gurnish_cases/Good/I/CaseI_centerline_with_metrics.vtp",
        'fig_save_folder': 'outputs_ber',
        'case_name': 'CaseI',
        'gurnish_case_name': 'Case I individual'
    },
}

class Bernoulli():
    def __init__(self, case_dict):
        self.Q = case_dict["inlet_flow_rate"]
        self.cline_file = case_dict["cline"]
        self.fig_save_folder = case_dict["fig_save_folder"]
        self.case_name = case_dict["case_name"]
        self.gurnish_case_name = case_dict["gurnish_case_name"] #Case name Gurnish uses to distinguish the case in her tecplot data
        self.density = 1.06 #g/mL
        self.K = 1.5 #Empirically derived constant

        self.get_values()

    def get_bernoulli_data(self):
        with open("Gurnish_Data/Gurnish_Case_Data.pkl", "rb") as f:
            df = pickle.load(f)

        case_name = self.gurnish_case_name
        x = df[case_name]["dist"]
        pber = df[case_name]["pber"]
        pcen = df[case_name]["pcen"]
        return x, pber, pcen

    def get_values(self):
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(self.cline_file)
        reader.Update()
        polydata = reader.GetOutput()
            
        self.area_array_np = vtk_to_numpy(polydata.GetPointData().GetArray("CrossSectionArea")) / 100 #Units of cm^2

        points = vtk_to_numpy(polydata.GetPoints().GetData())
        diffs = np.diff(points, axis=0)
        seg_lens = np.linalg.norm(diffs, axis=1)
        self.length_array_np = np.concatenate([[0.0], np.cumsum(seg_lens)]) 
        self.length_array_np /= 10 #Converting from units of mm^2 -> cm^2
        
    ########################
    ##### Calculations #####
    ########################

    def calculate_velocities(self):
        areas = self.area_array_np.copy()
        V = np.zeros(len(areas))
        for i in range(len(areas)):
            V[i] = (self.Q / areas[i]) * 1.5
        print(f"Max Velocity: {max(V)}")
        print(f"Max Velocity Index: {np.argmax(V)}")
        self.V = V

    def calculate_pressure_drop(self):
        areas = self.area_array_np.copy()
        delta_p = np.zeros(len(areas))
        total_p_drop = np.zeros(len(areas))
        V_1_squared = self.V[0] ** 2
        # V_1_squared = 0

        for i in range(1, len(areas)):
            delta_p[i] = (-0.5 * self.density * (self.V[i]**2 - V_1_squared)) / 1333.2 #Calculating the instantaneous pressure drop at that point - in mmHg
            if delta_p[i] < delta_p[i-1]:
                total_p_drop[i] = total_p_drop[i-1] + (delta_p[i] - delta_p[i-1])  #Accumulated pressure drop
            else:
                total_p_drop[i] = total_p_drop[i-1]
        total_p_drop[-1] = total_p_drop[-2]
        print(f"Max delta_p: {min(delta_p)}")
        print(f"Max index delta p: {np.argmin(delta_p)}")
        self.delta_p = delta_p
        self.total_p_drop = total_p_drop

    def accumulate_gurnish_data(self, pber):
        total_pber = np.zeros(len(pber))
        for i in range (1, len(pber)):
            if pber[i] < pber[i-1]:
                total_pber[i] = total_pber[i-1] + (pber[i] - pber[i-1])
            else:
                total_pber[i] = total_pber[i-1]

        return total_pber

    def plot(self):
        #Getting true values from Gurnish's data
        x_true, pber_gurnish, pcen_cfd = self.get_bernoulli_data()
        #Get the calculated data
        x = self.length_array_np
        y = self.total_p_drop

        #Get Gurnish Accumulated data
        acc_pber_gur = self.accumulate_gurnish_data(pber_gurnish)

        #Creating the plot
        fig, ax = plt.subplots(1, 1, figsize=(10,6))
        ax.plot(x, y, color="orange", linewidth=2, label="Calculated - Accumulated")
        ax.plot(x_true, acc_pber_gur, color="purple", linewidth=2, label="Gurnish")
        ax.plot(x_true, pcen_cfd, color="black", linewidth=2, label="CFD")
        ax.set_title(f"{self.case_name}", fontsize=16)
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"outputs_bernoulli/{self.case_name}.png", dpi=300)
        plt.show()

    def metrics(self):
        areas = self.area_array_np.copy()
        area_0 = areas[0]
        min_area = min(areas)

        print(f"Minimum stenotic area: {min_area}\tIndex: {np.argmin(areas)}\n")
        print(f"Proximal area: {area_0}")

    def run(self):
        self.calculate_velocities()
        self.calculate_pressure_drop()

        self.plot()
        self.metrics()

def main():
    case = "CaseA"
    Ber = Bernoulli(PRESET_CASES[case])
    Ber.run()

if __name__ == "__main__":
    main()