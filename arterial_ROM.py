'''
Trying to implement the method from this paper to see if it works for our cerebral venous cases
https://doi.org/10.1371/journal.pone.0258047
'''

import numpy as np
import pandas as pd
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d
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

class ROM():
    def __init__(self, case_dict, ):
        self.Q = case_dict["inlet_flow_rate"]
        self.cline_file = case_dict["cline"]
        self.fig_save_folder = case_dict["fig_save_folder"]
        self.case_name = case_dict["case_name"]
        self.gurnish_case_name = case_dict["gurnish_case_name"] #Case name Gurnish uses to distinguish the case in her tecplot data
        self.density = 1.06 #g/mL
        self.K = 1.5 #Empirically derived constant
        self.dyn_visc = 0.037

    def get_values(self):
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(self.cline_file)
        reader.Update()
        polydata = reader.GetOutput()
            
        self.radius_array_np = vtk_to_numpy(polydata.GetPointData().GetArray("HydraulicDiameter")) / 2 #units of mm
        self.area_array_np = vtk_to_numpy(polydata.GetPointData().GetArray("CrossSectionArea")) #Units of mm^2

        points = vtk_to_numpy(polydata.GetPoints().GetData())
        diffs = np.diff(points, axis=0)
        seg_lens = np.linalg.norm(diffs, axis=1)
        self.length_array_np = np.concatenate([[0.0], np.cumsum(seg_lens)])

    def calculate_pressures(self):
        areas = self.area_array_np.copy()
        A_0 = areas[0]
        R_0 = self.radius_array_np[0]
        D_0 = R_0 * 2
        dps = np.zeros(len(areas))
        for i in range(len(areas)):
            A_s = areas[i]
            D_s = self.radius_array_np[i] * 2
            delta_p = ((self.dyn_visc * self.Q) / (2*np.pi*R_0**3)) + (2 * (self.dyn_visc / self.density) * ((1/A_s - 1/A_0)**2)*self.Q**3 / (self.density * (1 - D_s/D_0)))
            dps[i] = delta_p

        self.pressures = dps

    def plot(self):
        pass

    def run(self):
        pass

def main():
    ReducedOrderModel = ROM()
    ReducedOrderModel.run()

if __name__ == "__main__":
    main()