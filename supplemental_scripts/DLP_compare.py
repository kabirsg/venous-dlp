'''
Trying to compare adding different DLP contributions to see how it aligns to CFD results
'''

import numpy as np 
import pandas as pd
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d
from datetime import datetime
import pickle

PRESET_CASES = {
    'CaseA': {
        'inlet_flow_rate': 3.73,
        'cline': "/home/kabir/masters_files/Gurnish_cases/Good/A/CaseA_centerline_with_metrics.vtp",
        'fig_save_folder': '../outputs',
        'case_name': 'CaseA',
        'cfd_case_name': 'Case A individual'
    },
    'CaseC': {
        'inlet_flow_rate': 5.00,
        'cline': "/home/kabir/masters_files/Gurnish_cases/Good/C/CaseC_centerline_with_metrics.vtp",
        'fig_save_folder': '../outputs',
        'case_name': 'CaseC',
        'cfd_case_name': 'Case C individual'
    },
    'CaseE': {
        'inlet_flow_rate': 3.48,
        'cline': "/home/kabir/masters_files/Gurnish_cases/Good/E/CaseE_centerline_with_metrics.vtp",
        'fig_save_folder': '../outputs',
        'case_name': 'CaseE',
        'cfd_case_name': 'Case E individual'
    },
    'CaseF': {
        'inlet_flow_rate': 7.93,
        'cline': "/home/kabir/masters_files/Gurnish_cases/Good/F/CaseF_centerline_with_metrics.vtp",
        'fig_save_folder': '../outputs',
        'case_name': 'CaseF',
        'cfd_case_name': 'Case F individual'
    },
    'CaseG': {
        'inlet_flow_rate': 5.18,
        'cline': "/home/kabir/masters_files/Gurnish_cases/Good/G/CaseG_centerline_with_metrics.vtp",
        'fig_save_folder': '../outputs',
        'case_name': 'CaseG',
        'cfd_case_name': 'Case G individual'
    },
    'CaseH': {
        'inlet_flow_rate': 4.50,
        'cline': "/home/kabir/masters_files/Gurnish_cases/Good/H/CaseH_centerline_with_metrics.vtp",
        'fig_save_folder': '../outputs',
        'case_name': 'CaseH',
        'cfd_case_name': 'Case H individual'
    },
    'CaseI': {
        'inlet_flow_rate': 5.40,
        'cline': "/home/kabir/masters_files/Gurnish_cases/Good/I/CaseI_centerline_with_metrics.vtp",
        'fig_save_folder': '../outputs',
        'case_name': 'CaseI',
        'cfd_case_name': 'Case I individual'
    },
}

class Compare():
    def __init__(self, case_dict):
        self.Q = case_dict["inlet_flow_rate"]
        self.cline_file = case_dict["cline"]
        self.fig_save_folder = case_dict["fig_save_folder"]
        self.case_name = case_dict["case_name"]
        self.cfd_case_name = case_dict["cfd_case_name"] #Case name Gurnish uses to distinguish the case in her tecplot data
        self.density = 1.06 #g/mL
        self.dyn_viscosity = 0.037 #dynamic viscosity mu value [Poise]
        self.K = 1.5 #Empirically derived constant
        self.Kt = 1.52 #Constant from Mirramezani paper

        print(f"Case: {self.case_name}")
        self.get_values()

    def get_bernoulli_data(self):
        with open("../Gurnish_Data/Gurnish_Case_Data.pkl", "rb") as f:
            df = pickle.load(f)

        case_name = self.cfd_case_name
        x = df[case_name]["dist"] #Distance along centerline
        pber = df[case_name]["pber"] #Bernoulli data
        pcen = df[case_name]["pcen"] #CFD data

        pber_acc = self.accumulate_gurnish_data(pber)
        return x, pber_acc, pcen

    '''
    Not in use! Or rather, it's just useless
    '''
    def accumulate_gurnish_data(self, pber):
        total_pber = np.zeros(len(pber))
        for i in range (1, len(pber)):
            if pber[i] < pber[i-1]:
                total_pber[i] = total_pber[i-1] + (pber[i] - pber[i-1])
            else:
                total_pber[i] = total_pber[i-1]

        return total_pber

    '''
    Calculates the Reynold's number for each centerline point
    
    Returns:
        - reynolds: List of the Reynold's number for all points. [Unitless]
    '''
    def create_reynolds_array(self):
        reynolds = []
        areas= self.area_array_np.copy()
        radii = self.radius_array_np.copy()
        flow_rate = self.Q
        dyn_visc = self.dyn_viscosity
        density = self.density
        for i, rad in enumerate(radii):
            re = ((flow_rate * density / areas[i]) * (rad*2)) / dyn_visc #Re = ((Q/CSA)*Dh)/dynamic viscosity 
            reynolds.append(re)
        return reynolds

    def get_values(self):
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(self.cline_file)
        reader.Update()
        polydata = reader.GetOutput()

        self.radius_array_np = vtk_to_numpy(polydata.GetPointData().GetArray("HydraulicDiameter")) / 20 #Diameter -> Radius, mm -> cm
        self.area_array_np = vtk_to_numpy(polydata.GetPointData().GetArray("CrossSectionArea")) / 100 #Units of cm^2

        points = vtk_to_numpy(polydata.GetPoints().GetData())
        diffs = np.diff(points, axis=0)
        seg_lens = np.linalg.norm(diffs, axis=1)
        self.seg_lens_array_np = seg_lens.copy() / 10 #mm -> cm
        self.length_array_np = np.concatenate([[0.0], np.cumsum(seg_lens)]) 
        self.length_array_np /= 10 #Converting from units of mm -> cm

        self.curvature_array_np = 1 / vtk_to_numpy(polydata.GetPointData().GetArray("Curvature"))

        self.re_array = self.create_reynolds_array()

    ########################
    ##### Calculations #####
    ########################

    def calculate_velocities(self, areas):
        V = np.zeros(len(areas))
        for i in range(len(areas)):
            V[i] = (self.Q / areas[i]) * 1.5
        print(f"Max Velocity: {max(V)}")
        print(f"Max Velocity Index: {np.argmax(V)}")
        self.V = V

    def bernoulli(self):
        areas = self.area_array_np.copy()
        delta_p = np.zeros(len(areas))
        total_p_drop = np.zeros(len(areas))

        #Handle Vs
        self.calculate_velocities(areas)
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
        self.ber_delta_p = delta_p
        self.ber_p_drop = total_p_drop

    def calculate_viscous_resistance(self):
        visc_res = np.zeros(len(self.length_array_np))
        CONST_TERM = 8 * self.dyn_viscosity / np.pi

        #Actually calculating the viscous resistance
        visc_res[0] = CONST_TERM * (self.seg_lens_array_np[0]/2) / (self.radius_array_np[0] ** 4)
        for i in range(1, len(self.length_array_np)-1):
            L_i = (self.seg_lens_array_np[i-1] + self.seg_lens_array_np[i]) / 2
            visc_res[i] = CONST_TERM * L_i / (self.radius_array_np[i] ** 4)
        visc_res[-1] = CONST_TERM * (self.seg_lens_array_np[-1]/2) / (self.radius_array_np[-1] ** 4)

        self.visc_pressure = self.Q * visc_res / 1333.2 #Viscous pressure drop in mmHg

    def calculate_viscous_resistance_with_curvature(self):
        visc_res = np.zeros(len(self.length_array_np))
        CONST_TERM = 8 * self.dyn_viscosity / np.pi
        K_i = self.re_array[0] * np.sqrt(self.radius_array_np[0] / self.curvature_array_np[0])
        curv = 0.1033 * np.sqrt(K_i) * ((1+(1.729 / K_i)) ** 0.5 - (1.315 / np.sqrt(K_i))) ** -3
        visc_res[0] = (CONST_TERM * (self.seg_lens_array_np[0]/2) * curv) / (self.radius_array_np[0]**4)
        for i in range(1, len(self.area_array_np)-1):
            K_i = self.re_array[i] * np.sqrt(self.radius_array_np[i] / self.curvature_array_np[i])
            curv = 0.1033 * np.sqrt(K_i) * ((1+(1.729 / K_i)) ** 0.5 - (1.315 / np.sqrt(K_i))) ** -3

            multiplier = max(curv, 1e-8)

            L_i = self.length_array_np[i-1]/2 + self.length_array_np[i]/2
            visc_res[i] = (CONST_TERM * L_i * multiplier) / (self.radius_array_np[i]**4)

        visc_res[-1] = (CONST_TERM * (self.seg_lens_array_np[-1]/2) * curv) / (self.radius_array_np[-1]**4)
        self.visc_curv_pressure = self.Q * visc_res / 1333.2

    def calculate_expansion_resistance(self):
        def calculate_added_resistance(A_s, A_0):
            return ((self.density * self.Kt/(2*(A_0**2))) * ((A_0/A_s) - 1) ** 2) * abs(self.Q)

        def distribute_expansion_resistance(min_idx, val, resistances, max_indices):
            #Find the next maximum after this local minimum
            next_max = max_indices[max_indices > min_idx]
            if len(next_max) == 0:
                #If there is no downstream maximum - apply entirely at the minimum point
                resistances[min_idx] += val
                return resistances
            next_max_idx = next_max[0]
    
            #Points in the recovery region (inclusive of both endpoints)
            region_indices = list(range(min_idx, next_max_idx + 1))
    
            #Equal share per point
            r_per_point = val / len(region_indices)
            for idx in region_indices:
                resistances[idx] += r_per_point
            
            return resistances

        def create_min_max_array(self):
                radii = self.radius_array_np.copy()
                maxima_indices, _ = find_peaks(radii)
                minima_indices, _ = find_peaks(-radii)
        
                all_indices = np.concatenate([maxima_indices, minima_indices])
                all_indices.sort()
        
                regions = [] #List of lists containing the expansion regions (by index)
                curr_region = [] #List that will contain two elements: Min and maximum for the stenotic region
                first_max = 0 #Keeps track of where the first local maximum is
        
                #Error checking
                if minima_indices[0] > maxima_indices[-1]:
                    print("WARNING: NO EXPANSION REGION - CONTINUING AS NORMAL AND HOPING FOR THE BEST (THIS IS UNTESTED BEHAVIOUR)")
                    return [[]]
        
                if maxima_indices[0] < minima_indices[0]:
                    first_max = maxima_indices[0]
        
                #Iterating over every extrema point from the first minimum to the last maximum
                for i in range(minima_indices[0], maxima_indices[-1]+1):
                    if i in maxima_indices:
                        if len(curr_region) == 1:
                            #Minimum added, adding maximum
                            curr_region.append(i)
                        elif len(curr_region) == 2:
                            #Two local maximums back to back
                            curr_region[1] = i
                        else:
                            #Vessel starts with a local maximum - This should never be hit
                            continue
                    elif i in minima_indices:
                        if len(curr_region) == 2:
                            regions.append(curr_region)
                        curr_region = [i] #new expansion region starting from minimum
                    else:
                        #Not a maximum or minimum point
                        continue
        
                return regions, first_max

        exp_res_dict = {}
        exp_regions, first_max = create_min_max_array(self)

        #Handle the first expansion region
        A_0 = (self.area_array_np[first_max] + self.area_array_np[exp_regions[0][1]])
        A_s = self.area_array_np[exp_regions[0][0]]
        delta_R = calculate_added_resistance(A_s, A_0)
        exp_res_dict[exp_regions[0][0]] = delta_R
        #Handle the rest of them
        for i in range(1, len(exp_regions)):
            if len(exp_regions[i]) != 2:
                print("There is an error here with the array containing the start and end points of exp region: ", exp_regions[i])
                print(exp_regions)
            A_0 = (self.area_array_np[exp_regions[i-1][1]] + self.area_array_np[exp_regions[i][1]])/2
            A_s = self.area_array_np[exp_regions[i][0]]

            delta_R = calculate_added_resistance(A_s, A_0)
            exp_res_dict[exp_regions[i][0]] = delta_R

        max_indices, _ = find_peaks(self.radius_array_np)
        exp_resistances = np.zeros(len(self.area_array_np))
        for key, val in exp_res_dict.items():
            exp_resistances = distribute_expansion_resistance(key, val, exp_resistances, max_indices)

        self.exp_pressures = exp_resistances * self.Q / 1333.2 #in mmHg

    def combinations(self):
        ber = self.ber_p_drop.copy()
        vis = self.visc_pressure.copy()
        vis_curv = self.visc_curv_pressure.copy()
        exp = self.exp_pressures.copy()

        vce = vis_curv + exp
        vcb = vis_curv + ber
        vb = vis + ber

        return vce, vcb, vb

    ####################
    ##### PLOTTING #####
    ####################

    def plot(self):
        x = self.length_array_np
        vce, vcb, vb = self.combinations()
        x_gur, _, pcen = self.get_bernoulli_data()

        #Creating the plot
        fig, ax = plt.subplots(1, 1, figsize=(10,6))
        ax.plot(x_gur, pcen, color="black", linestyle="--", label="CFD")
        ax.plot(x, vce, color="red", label="Viscous Curvature Expansion")
        ax.plot(x, vcb, color="green", label="Viscous Curvature Bernoulli")
        ax.plot(x, vb, color="blue", label="Viscous Bernoulli")
        ax.set_title(f"{self.case_name} Pressures", fontsize=16)
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"../outputs_compare/{self.case_name}.png", dpi=300)
        plt.show()
            

    def run(self):
        self.bernoulli()
        self.calculate_viscous_resistance()
        self.calculate_viscous_resistance_with_curvature()
        self.calculate_expansion_resistance()

        self.plot()

def main():
    case = "CaseA"
    comparer = Compare(PRESET_CASES[case])
    comparer.run()

if __name__ == "__main__":
    main()