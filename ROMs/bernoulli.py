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
        # 'cline': "/home/kabir/masters_files/Gurnish_cases/Good/F/CaseF_centerline_with_metrics.vtp",
        'cline': "/home/kabir/masters_files/Gurnish_cases/Good/F/CaseF_centerline_reversed_with_metrics.vtp",
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
        # 'cline': "/home/kabir/masters_files/Gurnish_cases/Good/I/CaseI_centerline_with_metrics.vtp",
        'cline': '/home/kabir/masters_files/Gurnish_cases/Good/I/CaseI_centerline_reversed_with_metrics.vtp',
        'fig_save_folder': 'outputs_ber_rev',
        'case_name': 'CaseI_rev',
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
        print(f"Case: {self.case_name}")
        self.get_values()

    def get_values(self):
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(self.cline_file)
        reader.Update()
        polydata = reader.GetOutput()

        # self.area_array_np = vtk_to_numpy(polydata.GetPointData().GetArray("CrossSectionArea")) / 100
        self.area_array_np = vtk_to_numpy(polydata.GetPointData().GetArray("MaximumInscribedSphereRadius")) / 10 #Units of mm^2 -> cm^2
        self.area_array_np = self.area_array_np ** 2
        self.area_array_np *= np.pi
        # print(self.area_array_np)
        print(np.mean(self.area_array_np))

        points = vtk_to_numpy(polydata.GetPoints().GetData())
        diffs = np.diff(points, axis=0)
        seg_lens = np.linalg.norm(diffs, axis=1)
        self.length_array_np = np.concatenate([[0.0], np.cumsum(seg_lens)]) 
        self.length_array_np /= 10 #Converting from units of mm^2 -> cm^2

    def get_bernoulli_data(self):
        with open("Gurnish_Data/Gurnish_Case_Data.pkl", "rb") as f:
            df = pickle.load(f)

        case_name = self.gurnish_case_name
        x = df[case_name]["dist"]
        pber = df[case_name]["pber"]
        pcen = df[case_name]["pcen"]
        return x, pber, pcen

    def accumulate_gurnish_data(self, pber):
        total_pber = np.zeros(len(pber))
        for i in range (1, len(pber)):
            if pber[i] < pber[i-1]:
                total_pber[i] = total_pber[i-1] + (pber[i] - pber[i-1])
            else:
                total_pber[i] = total_pber[i-1]

        return total_pber
        
    ########################
    ##### Calculations #####
    ########################

    def calculate_velocities(self):
        areas = self.area_array_np.copy()
        print(np.mean(areas))
        V = np.zeros(len(areas))
        for i in range(len(areas)):
            V[i] = (self.Q / areas[i]) * 1.5
        print(f"Max Velocity: {max(V)}")
        print(f"Max Velocity Index: {np.argmax(V)}")
        self.V = V
        print(np.mean(V))

    def calculate_pressure_drop(self):
        areas = self.area_array_np.copy()
        delta_p = np.zeros(len(areas))
        total_p_drop = np.zeros(len(areas))
        # V_1_squared = self.V[0] ** 2
        V_1_squared = 0
        if V_1_squared == 0:
            self.case_name = f"{self.case_name}_0"

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

    ####################
    ##### PLOTTING #####
    ####################

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
        ax.plot(x, y, color="red", linewidth=2, label="Calculated - Accumulated")
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

    def check(self):
        save_path = f"outputs_bernoulli/{self.case_name}.xlsx"
        x_gur, pber_gur, _ = self.get_bernoulli_data()
        acc_pber_gur = self.accumulate_gurnish_data(pber_gur)
        dps = self.delta_p.copy()

        #Create dfs to put into excel file
        df_gur = pd.DataFrame({
            "x_gur": x_gur,
            "Gurnish Data": pber_gur,
            "Accumulated Gurnish": acc_pber_gur,
        })

        df_me = pd.DataFrame({
            "x_me": self.length_array_np,
            "Me Delta p": dps,
            "Me Accumulated": self.total_p_drop 
        })

        with pd.ExcelWriter(save_path) as writer:
            df_me.to_excel(writer, sheet_name="Me", index=True)
            df_gur.to_excel(writer, sheet_name="Gurnish", index=True)
        
        

    def run(self):
        self.calculate_velocities()
        self.calculate_pressure_drop()

        self.plot()
        self.metrics()
        self.check()

    def run_and_return(self):
        self.calculate_velocities()
        self.calculate_pressure_drop()

        x_true, pber_gurnish, pcen_cfd = self.get_bernoulli_data()
        acc_pber_gur = self.accumulate_gurnish_data(pber_gurnish)

        #Get the calculated data
        x = self.length_array_np
        y = self.total_p_drop

        return x, y, x_true, acc_pber_gur, pcen_cfd

def run_them_all():
    COLOURS = ["red", "blue", "green", "purple", "orange", "black", "gray"]
    fig, ax = plt.subplots(1, 1, figsize=(10,6))
    count = 0

    for case, case_dict in PRESET_CASES.items():
        colour = COLOURS[count]
        count += 1
        Ber = Bernoulli(case_dict)
        x, y, x_gur, acc_pber_gur, pcen_cfd = Ber.run_and_return()

        ax.plot(x, y, color=colour, label=f"Me: {case}")
        ax.plot(x_gur, acc_pber_gur, color=colour, linestyle=":", label="Gurnish")
        ax.plot(x_gur, pcen_cfd, color=colour, linestyle="--", label="CFD")

    ax.legend()
    ax.set_title("All Cases: My Bernoulli vs. Gurnish Bernoulli vs. CFD")
    plt.tight_layout()
    plt.savefig("outputs_bernoulli/all_cases.png", dpi=300)
    plt.show()


def main():
    case = "CaseF"
    if case == "all":
        run_them_all()
    else:
        Ber = Bernoulli(PRESET_CASES[case])
        Ber.run()

if __name__ == "__main__":
    main()