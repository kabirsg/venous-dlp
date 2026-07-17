import numpy as np 
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from datetime import datetime
import pandas as pd
import pickle

class LumpedParameter:
    def __init__(self, cline_file, Q, rho, Kt, mu, exp, fig_save_folder, case_name, inlet_point_idx=0):
        self.centerline_file = cline_file
        self.flow_rate = Q
        self.density = rho
        self.Kt = Kt
        self.dyn_viscosity = mu
        self.expansion = exp
        self.figure_save_folder = fig_save_folder
        self.inlet_point_idx = inlet_point_idx
        self.case_name = case_name

        self.no_visc_in_exp = True

        #Creating the polydata object
        if not Path(self.centerline_file).exists(): 
            raise FileNotFoundError("The centerline file at the path specified could not be found. Please double check the path provided")

        self.create_polydata()
        self.create_arrays() #Get the data from the centerline file

    '''
    Creating the polydata reader object that other class functions can use
    '''
    def create_polydata(self):
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(self.centerline_file)
        reader.Update()
        self.polydata = reader.GetOutput()

    '''
    Accurately calculate the length array, even if the inlets and outlets were not labelled correctly. 
    The self.inlet_point_idx variable determines where the inlet point is
    '''
    def create_length_array(self):
        diffs = np.diff(self.point_array_np, axis=0)
        seg_lengths = np.linalg.norm(diffs, axis=1)
        cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
        if self.inlet_point_idx is None:
            ref = cumulative[0]
        else:
            ref = cumulative[self.inlet_point_idx]
        length_array = np.abs(cumulative - ref) / 10
        return length_array

    '''
    Creates an array which has the length of each segment that belongs to each centerline point.
    For a given point P_i, the segment length = 0.5*(dist(P_i-1, P_i) + dist(P_i, P_i+1))
    Must be called AFTER self.create_length_array function has been run - requires self.length_array variable to exist
    C: Handling of the first and last elements have to change
    '''
    def create_segments_array(self):
        seg_lens_array = []
        #Handling i = 0 (first centerline point)
        seg_lens_array.append(self.length_array[1]/2)

        #For the middle elements
        for i in range(1, len(self.length_array)-1):
            back_L = self.length_array[i] - self.length_array[i-1]
            forward_L = self.length_array[i+1] - self.length_array[i]
            seg_length = back_L/2 + forward_L/2
            seg_lens_array.append(seg_length)

        #Handle i = -1 case (last centerline point)
        seg_lens_array.append((self.length_array[-1] - self.length_array[-2])/2)

        return seg_lens_array

    '''
    Calculates the Reynold's number for each centerline point
    
    Returns:
        - reynolds: List of the Reynold's number for all points. [Unitless]
    '''
    def create_reynolds_array(self):
        reynolds = []
        areas= self.area_array_np.copy()
        radii = self.radius_array_np.copy()
        flow_rate = self.flow_rate
        dyn_visc = self.dyn_viscosity
        density = self.density
        for i, rad in enumerate(radii):
            re = ((flow_rate * density / areas[i]) * (rad*2)) / dyn_visc #Re = = ((Q/CSA)*Dh)/dynamic viscosity 
            reynolds.append(re)
        return reynolds
    '''
    Creating arrays for the other class functions to use
    Created arrays:
        - self.radius_array_np - Numpy array containing Maximum Inscribed Sphere Radius at every centerline point [Units = cm]
        - self.point_array_np - Numpy array containing Location of every centerline point [Units = cm]
        - self.curvature_array_np - Numpy array containing Curvature at every centerline point [Units = 1/cm]
        - self.length_array - Numpy array containing the length along the centerline for every point [Units = cm]
    '''
    def create_arrays(self):
        # self.radius_array_np = vtk_to_numpy(self.polydata.GetPointData().GetArray("MaximumInscribedSphereRadius"))
        self.radius_array_np = vtk_to_numpy(self.polydata.GetPointData().GetArray("HydraulicDiameter")) #Using hydraulic diameter instead of MISR
        self.radius_array_np /= 10 #Adjusting for units: mm -> cm
        self.hyd_dia_array_np = self.radius_array_np.copy()
        self.radius_array_np /= 2 #Adjusting for hydraulic diameter -> hydraulic radius

        self.area_array_np = vtk_to_numpy(self.polydata.GetPointData().GetArray("CrossSectionArea"))
        self.area_array_np /= 100 #Adjusting for units: mm^2 -> cm^2
        
        self.perimeter_array_np = vtk_to_numpy(self.polydata.GetPointData().GetArray("CrossSectionPerimeter"))
        self.perimeter_array_np /= 10 #Adjusting for units: mm -> cm
        
        self.point_array_np = vtk_to_numpy(self.polydata.GetPoints().GetData())
    
        self.curvature_array_np = vtk_to_numpy(self.polydata.GetPointData().GetArray("Curvature"))
        self.curvature_array_np *= 10 #Adjusting for units: 1/mm -> 1/cm
        self.curvature_array_np = 1/self.curvature_array_np #Converting this to the radius of curvature. Units: cm. Note: Will print a warning if division by zero but will make value infinity and move on

        self.length_array = self.create_length_array()
        self.seg_lens_array = self.create_segments_array()
        self.re_array = self.create_reynolds_array()

        #Reverses the centerline points if the inlets and outlets are inversed
        if self.inlet_point_idx == -1:
            self.point_array_np = self.point_array_np[::-1]
            self.curvature_array_np = self.curvature_array_np[::-1]
            self.length_array = self.length_array[::-1]
            self.radius_array_np = self.radius_array_np[::-1]
            self.area_array_np = self.area_array_np[::-1]

    '''
    Calculating the viscous resistance term
    
    As per the Mirramezani et al. (2020) paper the viscous resistance term is defined as (8*mu/pi) * INT_0_L(1/(R(x)^4) dx)

    This value is multiplied by the maximum between gamma and zeta with gamma being resistances from curvature effects and zeta being resistances from unsteady effects

    The value of the curvature multiplier (gamma) is defined as: 0.1033*sqrt(K)*((1 + (1.729/K)^0.5) - (1.315/sqrt(K)))^-3 for every point
    
    This makes the final viscous resistance that is used for the calculation to be the following
    
    Final Viscous Resistance calculation: R_v = (8*mu/pi) * INT_0_L(gamma * 1/(R(x)^4) dx)

    Units: Q = mL/s (cm^3/s), L = cm, R = cm, K = -, a = 1/cm, R_v = g/(s*cm^4)

    Created variable:
        - self.viscous_resistances: List of the viscous resistances calculated at every centerline point
    '''
    def calculate_viscous_resistances(self):
        self.viscous_resistances = [] #List for viscous resistances
        CONST_TERM = 8*self.dyn_viscosity/np.pi #The constant term in the viscous resistance equation

        multiplier_array = []
        visc_diss_array = []
        K_array = []

        #Resistance contribution of all the centerline points until and excluding the last point
        #The length (L) is half the distance from the last point to this point and half the distance from this point to the next
        #Not using the first or last point since their radius values are a little funky and they are in the flow extension region anyways
        for i in range(1, len(self.point_array_np)-1):
            #Calculate the distances between the points
            back_L = self.length_array[i] - self.length_array[i-1]
            forward_L = self.length_array[i+1] - self.length_array[i]
            #Calculate the length of the segment for this centerline point
            L_i = back_L/2 + forward_L/2
            
            '''
            Change for the backwards method 
            '''
            L_i = back_L
            #Getting the radius at this point
            rad = self.radius_array_np[i]
            
            #Calculating the curvature term - gamma
            curv = self.curvature_array_np[i]
            K_i = self.re_array[i] * np.sqrt(rad / curv)
            K_array.append(K_i)
            curve_res_i = 0.1033 * np.sqrt(K_i) * ((1+(1.729 / K_i)) ** 0.5 - (1.315 / np.sqrt(K_i))) ** -3 #Multiplier to add the curvature resistance term
            
            #The viscous resistance "multiplier" is the maximum of gamma and zeta
            multiplier = max(curve_res_i, 1e-8)
            multiplier_array.append(multiplier)

            #Calculate the viscous resistance at this centerline point
            visc_res = (CONST_TERM * L_i * multiplier) / (rad ** 4)
            self.viscous_resistances.append(visc_res)

            #Calculate the pure viscous dissipation for the purpose of debugging
            visc_diss = (CONST_TERM * L_i) / (rad ** 4)
            visc_diss_array.append(visc_diss)

        self.multiplier_array = multiplier_array
        self.visc_dis_array = visc_diss_array
        self.Ks_array = K_array
        # print(f"Total Viscous Resistance: {sum(self.viscous_resistances)}")
        # print(f"Average of the multipliers: {sum(multiplier_array)/len(multiplier_array):.10f}")

    '''
    Creating the arrays for the local minimum and local maximum indices

    Method used:
        - argrelextrema: Scipy method finding the local minimum/maximum
            -order = 3: For each point, looks at the 3 points upstream and downstream to determine local maximum/minimum.
            Order gets reduced if there is a significant difference between the number of mins and maxs 
    
    Return:
        - minima_indices: Numpy array containing the indices for each of the local minimum
        - maxima_indices: Numpy array containing the indices for each of the local maximum
        - start_min: Boolean - True if there is a local minimium before a local maximum, False if local max before local min
    '''
    def create_min_max_array(self):
        accept = False
        order = 1
        while accept == False:
            minima_indices = argrelextrema(self.radius_array_np, np.less, order=order)[0] #Gets the inidices of the local minima - order = 3 means that 3 points on each side used for comparison to reduce noise
            maxima_indices = argrelextrema(self.radius_array_np, np.greater, order=order)[0]
            #start_min = minima_indices[0] < maxima_indices[0] #True if the index of the first minima is less than the index of the first maxima

            if abs(len(minima_indices) - len(maxima_indices)) < 2:
                accept = True
            else:
                order -= 1
                if order == 0:
                    raise ValueError("Need to be able to handle subsequent extrema value being of the same type for this case to work.")

        return minima_indices, maxima_indices

    '''
    Helper function for calculate_expansion_resistances function (below)
    
    Does the actual calculation for calculating the expansion resistance, given the Areas

    Parameters:
        - A_s: Cross sectional area at the local minimum
        - A_0: Mean cross setional area of surrounding local maximum

    Return:
        - Calculated expansion resistance
    '''
    def calculate_added_resistance(self, A_s, A_0):
        try:
            return ((self.density * self.Kt/(2*(A_0**2))) * ((A_0/A_s) - 1) ** 2) * abs(self.flow_rate)
        except Exception as e:
            print(f'Exception encountered: {e}')
            return 0

    '''
    Function to calculate the expansion resistance term.
    Has a different flow based on if the first local extrema point is a local maximum or local minimum

    Uses the following values:
        - create_min_max_array to get the lists of the local maxima and local minima

    Created class variables:
        - self.expansion_resistances: Numpy float of the total expansion resistance
        - self.exp_res_dict: Dictionary of the expansion resistance at every point
    '''
    def calculate_expansion_resistances(self):
        min_indices, max_indices = self.create_min_max_array()
        #List of the local min, max, min, max, etc. This works because they are always going to alternate min, max, etc.
        extrema_array = np.sort(np.concatenate((min_indices, max_indices)))
        exp_res_dict = {} #Empty for now - Eventually, Index : expansion pressure drop
        expansion_resistance = 0.0
        first = "min" if extrema_array[0] == min_indices[0] else "max"
        last = "min" if extrema_array[-1] == min_indices[-1] else "max"

        #If the first element is a maximum, no issues
        #If it's a minimum
        if first == "min":
            A_0 = self.area_array_np[extrema_array[0]]
            A_s = self.area_array_np[extrema_array[1]]

            delta_R = self.calculate_added_resistance(A_s, A_0)
            exp_res_dict[min_indices[0]] = delta_R
            expansion_resistance += delta_R

            #The first and last values aren't handled by the for loop
            for i in range(1, len(min_indices)-1):
                extrema_i = np.where(extrema_array == min_indices[i])[0][0]
                A_s = self.area_array_np[min_indices[i]]
                A_0 = (self.area_array_np[extrema_array[extrema_i-1]] + self.area_array_np[extrema_array[extrema_i+1]]) / 2
                # print(f"For region {min_indices[i]} to {max_indices[i]}, A_s: {A_s}, A_0: {A_0}")
                # print(f"For index {extrema_array[extrema_i-1]}, CSA: {self.area_array_np[extrema_array[extrema_i-1]]}")
                # print(f"For index {extrema_array[extrema_i]}, CSA: {self.area_array_np[extrema_array[extrema_i]]}")
                # print(f"For index {extrema_array[extrema_i+1]}, CSA: {self.area_array_np[extrema_array[extrema_i+1]]}")
                delta_R = self.calculate_added_resistance(A_s, A_0)
                # print(f'{extrema_array[extrema_i-1]} -> {min_indices[i]} -> {extrema_array[extrema_i+1]}: {delta_R}')
                exp_res_dict[min_indices[i]] = delta_R
                expansion_resistance += delta_R
            
            #Only have to deal with the last local extrema if it's a maximum. If it's a minimum, no expansion region.
            if last == "max":
                # print("Last is max")
                A_s = self.area_array_np[extrema_array[-2]]
                A_0 = (self.area_array_np[extrema_array[-3]] + self.area_array_np[extrema_array[-1]]) / 2
                delta_R = self.calculate_added_resistance(A_s, A_0)
                # print(f'{min_indices[-1]}: {delta_R}')
                exp_res_dict[min_indices[i]] = delta_R
                expansion_resistance += delta_R
            
        #If the first element is a maxima, need to treat a little differently
        else:
            for i in range(0, len(min_indices)-1):
                extrema_i = np.where(extrema_array == min_indices[i])[0][0]
                A_s = self.area_array_np[min_indices[i]]
                A_0 = (self.area_array_np[extrema_array[extrema_i-1]] + self.area_array_np[extrema_array[extrema_i+1]]) / 2
                # print(f"For region {max_indices[i]} to {min_indices[i]}, A_s: {A_s}, A_0: {A_0}")
                
                delta_R = self.calculate_added_resistance(A_s, A_0)
                # print(f'{min_indices[i]}: {delta_R}')
                exp_res_dict[min_indices[i]] = delta_R
                expansion_resistance += delta_R

            #Only having to deal with the last point if it's a maximum
            if last == "max":
                # print("Last element is a maximum")
                A_s = self.area_array_np[extrema_array[-2]]
                A_0 = (self.area_array_np[extrema_array[-3]] + self.area_array_np[extrema_array[-1]]) / 2

                delta_R = self.calculate_added_resistance(A_s, A_0)
                # print(f'{min_indices[-1]}: {delta_R}')
                exp_res_dict[min_indices[-1]] = delta_R
                expansion_resistance += delta_R

        self.expansion_resistances = expansion_resistance
        self.exp_res_dict = exp_res_dict
        # print(f'Total expansion resistance: {(expansion_resistance)}')

    '''
    Adding expansion resistance in the expansion resistance (from the local minimum to the downstream local maximum) 
    proportional to the radius at each point.

    Parameters:
        - key: The id of the local minimum (index in lists)
        - val: Total expansion resistance to be applied over the expansion region
        - resistances: List of previously calculated resistance values for every point
        - max_indices: List of the local maximum indices
    
    Return
        - resistances: List of calculated resistance values for every point after expansion resistance added
    '''
    def add_proportional_expansion_resistance(self, key, val, resistances, max_indices):
        #Find the next maximum after this local minimum
        next_max = max_indices[max_indices > key]
        if len(next_max) == 0:
            resistances[key] += val
            return resistances
        
        next_max_idx = next_max[0]
        region_indices = list(range(key, next_max_idx + 1))

        #Compute radius increase at each step in the region
        #Weight at point i = max(0, r[i] - r[i-1]), ie. only where expanding
        weights = []
        for idx in region_indices:
            if idx == key:
                weights.append(0.0)
            else:
                delta_r = self.radius_array_np[idx] - self.radius_array_np[idx - 1]
                weights.append(max(0.0, delta_r)) #Only positive growth counts
        
        total_weight = sum(weights)

        if total_weight == 0:
            #Flat or contraction region - fall back to equal distribution
            r_per_point = val / len(region_indices)
            for idx in region_indices:
                res_idx = idx - 1
                if 0 <= res_idx < len(self.viscous_resistances):
                    resistances[res_idx] = r_per_point
            
        else:
            for idx, w in zip(region_indices, weights):
                res_idx = idx - 1
                if 0 <= res_idx < len(self.viscous_resistances):
                    # print(f"At point: {res_idx}\tStarting resistance: {resistances[res_idx]}")
                    resistances[res_idx] += val * (w/total_weight)
                    # print(f"Final Resistance: {resistances[res_idx]}")

        return resistances
    
    def add_proportional_to_area_expansion_resisance(self, key, val, resistances, max_indices):
        #Find the next maximum after this local minimum
        next_max = max_indices[max_indices > key]
        if len(next_max) == 0:
            resistances[key] += val
            return resistances
        
        next_max_idx = next_max[0]
        region_indices = list(range(key, next_max_idx + 1))

        #Compute area increase at each step in the region
        #Weight at point i = max(0, a[i] - a[i-1]), ie. only where expanding
        weights = []
        for idx in region_indices:
            if idx == key:
                weights.append(0.0)
            else:
                delta_a = self.area_array_np[idx] - self.area_array_np[idx - 1]
                weights.append(max(0.0, delta_a)) #Only positive growth counts
        
        total_weight = sum(weights)

        if total_weight == 0:
            #Flat or contraction region - fall back to equal distribution
            r_per_point = val / len(region_indices)
            for idx in region_indices:
                res_idx = idx - 1
                if 0 <= res_idx < len(self.viscous_resistances):
                    resistances[res_idx] += r_per_point
            
        else:
            for idx, w in zip(region_indices, weights):
                res_idx = idx - 1
                if 0 <= res_idx < len(self.viscous_resistances):
                    resistances[res_idx] += val * (w/total_weight)
        
        return resistances

    '''
    Distributes the expansion losses in the post-stenotic expansion region proportional to the inverse of the radius.
    Note: This function applies loss AT the stenotic point as well as after, unlike the other distribution functions above.
    '''
    def add_exp_res_proportional_to_inverse_radius(self, min_idx, val, resistances, max_indices):
        #Find the next maximum after this local minimum
        next_max = max_indices[max_indices > min_idx]
        if len(next_max) == 0:
            resistances[min_idx] += val
            return resistances
        
        next_max_idx = next_max[0]

        #Compute radius increase at each step in the region
        #Weight at point i = inverse to the proportion of the total "radius" at that particular point
        total_weight = sum(self.radius_array_np[min_idx:next_max_idx])
        for i in range(min_idx, next_max_idx+1):
            weight = self.radius_array_np[i] / total_weight
            resistances[i] += (val * weight)

        print(sum(resistances))
        return resistances


    '''
    Calculating the pressure drop at every point from the calculated resistance values
    '''
    def calculate_pressures(self):
        self.pressure_drops_mmHg = [] #List of pressure drops due to resistances of each segment
        self.pressures_mmHg = [] #List of pressures at each point

        #Calculating Total resistance
        resistances = self.viscous_resistances.copy() #Viscous resistance term
        _, max_indices = self.create_min_max_array()

        #Adding expansion resistance
        for key, val in self.exp_res_dict.items():
            if self.expansion == 1:
                resistances = self.add_proportional_expansion_resistance(key, val, resistances, max_indices)
            elif self.expansion == 2:
                resistances = self.add_proportional_to_area_expansion_resisance(key, val, resistances, max_indices)
            else:
                resistances = self.add_exp_res_proportional_to_inverse_radius(key, val, resistances, max_indices)

        self.total_resistances = resistances

        pressure_mmHg = self.flow_rate * resistances[0] / 1333.2
        for resistance in resistances:
            delta_P = self.flow_rate * resistance #Pressure drop over each segment due to the resistive elements in that segment
            delta_P_mmHg = delta_P / 1333.22
            self.pressure_drops_mmHg.append(delta_P_mmHg)

            #Calculating new pressure
            pressure_mmHg -= delta_P_mmHg
            #Adding new pressure to the list of pressures at each point
            self.pressures_mmHg.append(pressure_mmHg)

        self.visc_pressures_mmHg = []
        self.exp_pressures_mmHg = []
        self.exp_resistances = []
        # visc_pressure = sum(self.viscous_resistances) * self.flow_rate / 1333.2
        # exp_pressure = sum(resistances) * self.flow_rate / 1333.2 - visc_pressure
        visc_pressure = 0
        exp_pressure = 0
        for i in range(len(resistances)):
            visc_res = self.viscous_resistances[i]
            exp_res = resistances[i] - self.viscous_resistances[i]
            self.exp_resistances.append(exp_res)

            dP_visc_mmHg = self.flow_rate * visc_res / 1333.22
            visc_pressure -= dP_visc_mmHg
            self.visc_pressures_mmHg.append(visc_pressure)

            dP_exp_mmHg = self.flow_rate * exp_res / 1333.22
            exp_pressure -= dP_exp_mmHg
            self.exp_pressures_mmHg.append(exp_pressure)
    
    '''
    Pressure calculation if there is no expansion resistance term (pure viscous resistive losses)
    '''
    def calculate_pressures_no_exp(self):
        self.pressure_drops_mmHg = [] #List of pressure drops due to resistances of each segment in mmHg
        self.pressures_mmHg = [] #List of pressures at each point in mmHg
        pressure_mmHg = 0

        resistances = self.viscous_resistances
        for resistance in resistances:
            delta_P = self.flow_rate * resistance #Pressure drop over each segment due to the resistive elements in that segment (dyn/cm^2)
            delta_P_mmHg = delta_P / 1333.2 #Pressure drop over each segment in mmHg

            #Calculating new pressure
            pressure_mmHg -= delta_P_mmHg

            #Adding to the list of pressures and pressure drops at each point
            self.pressures_mmHg.append(pressure_mmHg)
            self.pressure_drops_mmHg.append(delta_P_mmHg)
        
    '''
    Generating plot tracking the pressure pressure drops due to different sources
    '''
    def generate_pressure_drop_contributions_plots(self):
        # Create a figure with 1 row and 2 columns for side-by-side plots
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Using the same centerline slicing as your previous plots
        x = self.length_array[11:-11]
        
        # Slice the new pressure drop lists to match the length of x
        p_drop_visc = self.visc_pressures_mmHg[10:-10]
        p_drop_exp = self.exp_pressures_mmHg[10:-10]

        #Curvature
        dP_curv_array = np.array(self.viscous_resistances[10:-10]) - np.array(self.visc_dis_array[10:-10]) #Resistances just due to curvature
        dP_curv_array = dP_curv_array * (self.flow_rate / 1333.2) #Getting the pressure drop due to curvature [mmHg]
        
        #Viscous Dissipation
        dP_visc_diss_array = np.array(self.visc_dis_array[10:-10])
        dP_visc_diss_array = dP_visc_diss_array * (self.flow_rate / 1333.2)

        curv_pressure_array = []
        curv_pressure = 0
        visc_diss_array = []
        visc_diss = 0

        for i in range(len(dP_curv_array)):
            curv_pressure -= dP_curv_array[i]
            curv_pressure_array.append(curv_pressure)

            visc_diss -= dP_visc_diss_array[i]
            visc_diss_array.append(visc_diss)

        #Plotting 
        ax.plot(x, p_drop_visc, color='green', linewidth=2, label="Viscous Dissipation + Curvature - R_vc")
        ax.plot(x, p_drop_exp, color='blue', linewidth=2, label="Expansion - R_s")
        ax.plot(x, curv_pressure_array, color='red', linewidth=2, label="Curvature - R_c")
        ax.plot(x, visc_diss_array, color='purple', linewidth=2, label="Viscous Dissipation - R_v")
        ax.plot(x, self.pressures_mmHg[10:-10], color='black', linewidth=2, label="Total Pressure")

        ax.set_xlabel("Length Along Centerline (cm)", fontsize=16)
        ax.set_ylabel("Pressure Drop (mmHg)", fontsize=16)
        ax.set_title("Viscous vs Expansion Pressure Drop in LPM", fontsize=20)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        
        # Save and show the figure

        output_dir = Path(self.figure_save_folder)
        output_dir.mkdir(parents=True, exist_ok=True)

        save_path = f"{self.figure_save_folder}/{self.case_name}_pdrop_contributions_exp_{self.expansion}.png"
        plt.savefig(save_path, dpi=300)
        plt.show()

    def get_cfd_data(self):
        import config

        with open("Gurnish_Data/Gurnish_Case_Data.pkl", "rb") as f:
            zones_df = pickle.load(f)

        case_name = config.dlp_cfd_case_name

        x = zones_df[case_name]["dist"]
        pcen = zones_df[case_name]["pcen"]
        return x, pcen

    def get_bernoulli_data(self):
        import config
        with open("Gurnish_Data/Gurnish_Case_Data.pkl", "rb") as f:
            df = pickle.load(f)

        case_name = config.dlp_cfd_case_name
        x = df[case_name]["dist"]
        pber = df[case_name]["pber"]
        x_sten_idx = int(np.array(np.where(np.array(df[case_name]["diststen"] == 0)))[0][0])
        return x, pber, x_sten_idx

    '''
    Pressure Drop Plots with all sources and the total and a separate plot for CSA
    '''
    def plot_p_drops(self):
        # Using the same centerline slicing as your previous plots
        x = self.length_array[11:-11]
        
        # Slice the new pressure drop lists to match the length of x
        p_drop_visc = self.visc_pressures_mmHg[10:-10]
        p_drop_exp = self.exp_pressures_mmHg[10:-10]

        #Curvature
        dP_curv_array = np.array(self.viscous_resistances[10:-10]) - np.array(self.visc_dis_array[10:-10]) #Resistances just due to curvature
        dP_curv_array = dP_curv_array * (self.flow_rate / 1333.2) #Getting the pressure drop due to curvature [mmHg]
        
        #Viscous Dissipation
        dP_visc_diss_array = np.array(self.visc_dis_array[10:-10])
        dP_visc_diss_array = dP_visc_diss_array * (self.flow_rate / 1333.2)

        curv_pressure_array = []
        curv_pressure = 0
        visc_diss_array = []
        visc_diss = 0
        for i in range(len(dP_curv_array)):
            curv_pressure -= dP_curv_array[i]
            curv_pressure_array.append(curv_pressure)

            visc_diss -= dP_visc_diss_array[i]
            visc_diss_array.append(visc_diss)

        x2, cfd_data = self.get_cfd_data()
        x3 = self.length_array[1:-1]

        # Slice the new pressure drop lists to match the length of x - dP
        visc_delta_P = np.array(self.viscous_resistances.copy()) * self.flow_rate / 1333.2
        dP_exp_array = np.array(self.exp_resistances) * self.flow_rate / 1333.2
        
        #Curvature - dP
        dP_curv_array = np.array(self.viscous_resistances) - np.array(self.visc_dis_array) #Resistances just due to curvature
        dP_curv_array = dP_curv_array * (self.flow_rate / 1333.2) #Getting the pressure drop due to curvature [mmHg]

        #Viscous Dissipation - dP
        dP_visc_diss_array = np.array(self.visc_dis_array)
        dP_visc_diss_array = dP_visc_diss_array * (self.flow_rate / 1333.2)

        #Total dP
        dP_total = np.array(self.total_resistances)
        dP_total = dP_total * (self.flow_rate / 1333.2)

        #dP for CFD
        dP_cfd_data_array = [0]
        dP_cfd_data = cfd_data.copy()
        for i in range(1, len(dP_cfd_data)):
            dP_cfd_data_array.append(dP_cfd_data[i-1] - dP_cfd_data[i])

        #Bernoulli
        x4, pber, x_sten_idx = self.get_bernoulli_data()
        dP_ber_data_array = [0]
        for i in range(1, x_sten_idx+1):
            i_back_dP = pber[i-1] - pber[i]
            if i_back_dP >= -0.5:
                dP_ber_data_array.append(pber[i-1] - pber[i])

            else:
                # dP_ber_data_array.append(pber[i-1] - pber[i])
                break
        x5 = x4[:len(dP_ber_data_array)]

        dP_dict = {
            "x": x3,
            "Viscous Dissipation": dP_visc_diss_array,
            "Curvature": dP_curv_array,
            "Expansion": dP_exp_array,
            "Total": dP_total
        }
        bar_y = self.get_bar_plot_data(dP_dict)

        ############
        # Plotting #
        # ########## 
        fig, (ax1, ax2, ax3) = plt.subplots(3,1,figsize=(13,8))

        # ax1.plot(x, p_drop_visc, color='green', linewidth=2, label="Viscous Dissipation + Curvature - R_vc")
        ax1.plot(x, p_drop_exp, color='blue', linewidth=2, label="Expansion - R_s")
        ax1.plot(x, curv_pressure_array, color='green', linewidth=2, label="Curvature - R_c")
        ax1.plot(x, visc_diss_array, color='red', linewidth=2, label="Viscous Dissipation - R_v")
        ax1.plot(x, self.pressures_mmHg[10:-10], color='black', linewidth=1, label="Total Pressure")
        ax1.plot(x2, cfd_data, color='black', linewidth=2, linestyle="--", label="CFD Pressure")
        ax1.plot(x4, pber, color="purple", linewidth=1, label="Bernoulli")

        #CSA on twin axis
        ax1b = ax1.twinx()
        ax1b.plot(x3, self.area_array_np[1:-1], color="pink", linewidth=1, label="CSA")
        ax1b.tick_params(axis='y')
        ax1b.set_ylabel("Cross Sectional Area [cm^2]")

        # ax1.set_xlabel("Length Along Centerline (cm)")
        ax1.set_ylabel("Pressure Drop (mmHg)")
        ax1.set_title("Pressure Drop in LPM vs. CFD vs. Bernoulli", fontsize=16)
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.set_xlim(min(x3), max(x3))
        
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1b.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10)
        
        # ax2.plot(x, self.area_array_np[11:-11], color="blue", linewidth=2)
        # ax2.set_xlabel("Length Along Centerline [cm]", fontsize=10)
        # ax2.set_ylabel("Cross Sectional Area (CSA) [cm^2]", fontsize=10)
        #
        # ax2.plot(x3, visc_delta_P, color='green', linewidth=2, label="Viscous (Diss + Curv)")
        ax2.plot(x3, dP_exp_array, color='blue', linewidth=2)
        ax2.plot(x3, dP_curv_array, color='green', linewidth=2)
        ax2.plot(x3, dP_visc_diss_array, color="red", linewidth=2)
        ax2.plot(x3, dP_total, color='black', linewidth=1)
        ax2.plot(x2, dP_cfd_data_array, color='black', linewidth=2, linestyle="--")
        ax2.plot(x5, dP_ber_data_array, color='purple', linewidth=1)
        ax2.tick_params(axis='y')
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.set_xlim(min(x3), max(x3))

        # ax2.set_xlabel("Length Along Centerline (cm)")
        ax2.set_ylabel("Pressure Drop (mmHg)")

        ax2b = ax2.twinx()
        ax2b.plot(x3, self.area_array_np[1:-1], color="pink", linewidth=1, label="CSA")
        ax2b.tick_params(axis='y')

        ax2b.set_ylabel("Cross Sectional Area [cm^2]")
        ax2.set_title("Instantaneous Pressure Drops in LPM vs. CFD vs. Bernoulli", fontsize=16)
        ax2.grid(True, linestyle='--', alpha=0.7)

        ax3.stackplot(x3, bar_y, labels=['I', 'II', 'III'], colors=['red', 'green', 'blue'])
        ax3.set_ylim(0, 100)
        ax3.set_xlim(min(x3), max(x3))
        ax3.set_ylabel("% of Pressure Drop")
        ax3.set_xlabel("Length Along Centerline (cm)")
        ax3.set_title("Proportion of Pressure Drop due to Different Contributions for Each Point", fontsize=16)

        ax3b = ax3.twinx()
        ax3b.plot(x3, self.area_array_np[1:-1], color="pink", linewidth=1, label="CSA")
        ax3b.tick_params(axis='y')
        ax3b.set_ylabel("Cross Sectional Area [cm^2]")

        plt.tight_layout()
        
        # Save and show the figure
        output_dir = Path(self.figure_save_folder)
        output_dir.mkdir(parents=True, exist_ok=True)

        save_path = f"{self.figure_save_folder}/{self.case_name}_pdrop_w_bar_exp_{self.expansion}_Dh.png"
        plt.savefig(save_path, dpi=300)
        plt.show()

    def get_bar_plot_data(self, dP_dict):
        x = dP_dict["x"]
        v = dP_dict["Viscous Dissipation"]
        c = dP_dict["Curvature"]
        e = dP_dict["Expansion"]

        #Lists containing each loss as a percentage of the total
        v_p = []
        c_p = []
        e_p = []

        for i in range(len(x)):
            total_i = v[i] + c[i] + e[i]
            v_p.append(v[i] * 100 / total_i)
            c_p.append(c[i] * 100 / total_i)
            e_p.append(e[i] * 100 / total_i)

        v_p = np.array(v_p)
        c_p = np.array(c_p)
        e_p = np.array(e_p)

        return [v_p, c_p, e_p]

    # def bar_plot_v2(self, dP_dict):
    #     x_vals = dP_dict["x"]
    #     x = np.array(range(len(x_vals)))
    #     v = dP_dict["Viscous Dissipation"]
    #     c = dP_dict["Curvature"]
    #     e = dP_dict["Expansion"]

    #     #Lists containing each loss as a percentage of the total
    #     v_p = []
    #     c_p = []
    #     e_p = []

    #     for i in range(len(x)):
    #         total_i = v[i] + c[i] + e[i]
    #         v_p.append(v[i] * 100 / total_i)
    #         c_p.append(c[i] * 100 / total_i)
    #         e_p.append(e[i] * 100 / total_i)

    #     y = [v_p, c_p, e_p]

    #     fig, ax = plt.subplots(figsize=(10, 8))
    #     ax.stackplot(x, y, labels=['I', 'II', 'III'], colors=['red', 'green', 'blue'])

    #     ax.set_ylim(0, 100)
    #     ax.set_xlim(min(x), max(x))

    #     ax2 = ax.twinx()
    #     ax2.plot(x, self.area_array_np[1:-1], color="white", linewidth=1, label="CSA")
    #     ax2.tick_params(axis='y')
    #     ax2.set_ylabel("Cross Sectional Area [cm^2]")

    #     save_path = f"{self.figure_save_folder}/{self.case_name}_bar_plot_v2_exp_{self.expansion}.png"
    #     plt.savefig(save_path, dpi=300)
    #     plt.show()


    def excel_values(self):
        #0. Zero pad where necessary
        self.multiplier_array = [None] + self.multiplier_array + [None]
        self.Ks_array = [None] + self.Ks_array + [None]
        self.visc_dis_array = [0] + self.visc_dis_array + [0]
        self.viscous_resistances = [0] + self.viscous_resistances + [0]
        self.exp_resistances = [None] + self.exp_resistances + [None]
        self.total_resistances = [None] + self.total_resistances + [None]
        self.pressure_drops_mmHg = [None] + self.pressure_drops_mmHg + [None]
        self.pressures_mmHg = [None] + self.pressures_mmHg + [None]
        density_array = np.full(len(self.seg_lens_array), self.density)
        dyn_visc_array = np.full(len(self.seg_lens_array), self.dyn_viscosity)
        Kt_array = np.full(len(self.seg_lens_array), self.Kt)
        Q_array = np.full(len(self.seg_lens_array), self.flow_rate)

        # 1. Organize your data into a dictionary
        # This automatically handles the alignment of your data
        data = {
            "Point ID": np.arange(0, len(self.length_array)),
            "Distance Along Centerline [cm]": self.length_array,
            "Segment Length [cm]": self.seg_lens_array,
            "D_h [cm]": self.hyd_dia_array_np,
            "R_eff [cm]": np.sqrt(self.area_array_np / np.pi),
            "CSA [cm^2]": self.area_array_np,
            "Perimeter [cm]": self.perimeter_array_np,
            "Radius of Curvature [cm]": self.curvature_array_np,
            "gamma [-]": self.multiplier_array,
            "K - Dean's number [-]": self.Ks_array,
            "Re [-]": self.re_array,
            "Viscous Dissipation Only (Eq.3) [g/(s*cm^4)]": self.visc_dis_array,
            "Viscous Resistance w/ Curvature (Eq.6) - Viscous Dissipation Only (Eq.3) [g/(s*cm^4)]": np.array(self.viscous_resistances) - np.array(self.visc_dis_array),
            "Expansion Resistance (R_s) [g/(s*cm^4)]": self.exp_resistances,
            "Total Resistance (R) [g/(s*cm^4)]": self.total_resistances,
            "Pressure Drop (delta_P) [mmHg]": self.pressure_drops_mmHg,
            "Total Pressure Drop (P) [mmHg]": self.pressures_mmHg,
            "Density [g/cm^3]": density_array,
            "Dynamic Viscosity [Poise]": dyn_visc_array,
            "Kt [-]": Kt_array,
            "Inlet Flow Rate [mL/s]": Q_array
        }

        # 2. Create the DataFrame
        df = pd.DataFrame(data)

        # 3. Export to Excel
        excel_file_name = self.case_name
        df.to_excel(f"{excel_file_name}.xlsx", index=False)
        print(f"Excel file '{excel_file_name}.xlsx' has been generated.")

    '''
    Function to run everything in the correct order based on the parameter given during class initialization,
    to make this class easy to use.
    '''
    def run(self):

        if self.expansion > 3:
            raise ValueError("Expansion term value must be between 1 and 3")

        #Calculate viscous resistance
        self.calculate_viscous_resistances()

        #Calculate expansion resistance if necessary
        if self.expansion == 0:
            self.calculate_pressures_no_exp()
        else:
            self.calculate_expansion_resistances()
            self.calculate_pressures()
        
        # self.generate_pressure_drop_contributions_plots()
        self.plot_p_drops()

        self.excel_values()

def main():
    try:
        import config
        BLOOD_DYNAMIC_VISCOSITY = config.dlp_blood_dyn_visc
        INLET_FLOW_RATE = config.dlp_inlet_flow_rate
        KT = config.dlp_kt
        DENSITY = config.dlp_density

        EXPANSION = config.dlp_exp_term 
        try:
            INLET_POINT_IDX = config.dlp_inlet_point_idx
        except:
            INLET_POINT_IDX = 0 #If the user doesn't set an inlet point id, then assuming that the inlet is labelled as the first point

        CLINE_FILE_PATH = config.dlp_cline_file_path
        FIGURE_SAVE_FOLDER = config.dlp_fig_save_folder
        CASE_NAME = config.dlp_case_name

        
        
    except Exception as e:
        raise Exception(f"Please ensure that the config.py file is present in the same folder as this file and all the necessary variables are present: \n{e}")

    lp = LumpedParameter(
        cline_file=CLINE_FILE_PATH,
        Q=INLET_FLOW_RATE,
        rho=DENSITY,
        Kt=KT,
        mu=BLOOD_DYNAMIC_VISCOSITY,
        exp=EXPANSION,
        fig_save_folder=FIGURE_SAVE_FOLDER,
        case_name=CASE_NAME,

        inlet_point_idx=INLET_POINT_IDX
    )

    lp.run()

if __name__ == "__main__":
    main()