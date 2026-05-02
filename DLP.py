import numpy as np 
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from datetime import datetime
import pandas as pd

class LumpedParameter:
    def __init__(self, cline_file, Q, rho, Kt, mu, re, curv, exp, fig_save_folder, debug_options):
        self.centerline_file = cline_file
        self.flow_rate = Q
        self.density = rho
        self.Kt = Kt
        self.dyn_viscosity = mu
        self.reynolds = re
        self.curvature = curv
        self.expansion = exp
        self.figure_save_folder = fig_save_folder
        
        if debug_options:
            self.debug_file_path = debug_options[0]
            self.debug_case_name = debug_options[1]

        #Creating the polydata object
        if not Path(self.centerline_file).exists(): 
            raise FileNotFoundError("The centerline file at the path specified could not be found. Please double check the path provided")

        self.create_polydata()
        self.create_arrays() #Get the data from the centerline file

    '''
    Creating the polydata reader that other functions can use
    '''
    def create_polydata(self):
        reader = vtk.vtkXMLPolyDataReader() 
        reader.SetFileName(self.centerline_file)
        reader.Update()
        self.polydata = reader.GetOutput()

    '''
    Code from Rojin to more accurately calculate the length array, given that the inlet and outlet is messed up
    FIX: THIS SHOULDN'T HAVE A DEFAULT
    '''
    def create_length_array(self, inlet_point_id=1333):
        diffs = np.diff(self.point_array_np, axis=0)
        seg_lengths = np.linalg.norm(diffs, axis=1)
        cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
        self.length_array = np.abs(cumulative - cumulative[inlet_point_id]) / 10
        return

    '''
    Creating arrays for the other functions to use
    Created arrays:
        - self.radius_array_np - Numpy array containing Maximum Inscribed Sphere Radius at every centerline point [Units = cm]
        - self.point_array_np - Numpy array containing Location of every centerline point [Units = cm]
        - self.curvature_array_np - Numpy array containing Curvature at every centerline point [Units = 1/cm]
        - self.length_array - Numpy array containing the length along the centerline for every point [Units = cm]
    '''
    def create_arrays(self):
        self.radius_array_np = vtk_to_numpy(self.polydata.GetPointData().GetArray("MaximumInscribedSphereRadius"))
        self.radius_array_np /= 10 #Adjusting for units: mm -> cm 
        
        self.point_array_np = vtk_to_numpy(self.polydata.GetPoints().GetData())
    
        self.curvature_array_np = vtk_to_numpy(self.polydata.GetPointData().GetArray("Curvature"))
        self.curvature_array_np *= 10 #Adjusting for units: 1/mm -> 1/cm
        
        self.create_length_array(inlet_point_id=1333) #C: Fix - this shouldn't be hardcoded or even needed

        #This is needed because PTSeg028 has flipped inlets and outlets
        self.needs_flipping = True #C: This is weird I don't like this
        if self.needs_flipping:
            self.point_array_np = self.point_array_np[::-1]
            self.curvature_array_np = self.curvature_array_np[::-1]
            self.length_array = self.length_array[::-1]
            self.radius_array_np = self.radius_array_np[::-1]
    
    '''
    C: This is the Wormseley number, not the zeta unsteady value.
    '''
    def calculate_unsteady_term(self, rad):
        return rad * np.sqrt(self.density / self.dyn_viscosity)

    '''
    Calculating the viscous resistance term
    
    As per the Mirramezani et al. (2020) paper the viscous resistance term is defined as (8*mu/pi) * INT_0_L(1/(R(x)^4) dx)

    This value is multiplied by the maximum between gamma and zeta with gamma being resistances from curvature effects and zeta being resistances from unsteady effects

    The value of the curvature multiplier (gamma) is defined as: 0.1033*sqrt(K)*((1 + (1.729/K)^0.5) - (1.315/sqrt(K)))^-3 for every point
    The value of the unsteady multiplier (zeta) is defined as the Womersley number: zeta = alpha = R * sqrt(rho * frequency / mu) with frequency = 1
    
    This makes the final viscous resistance that is used for the calculation to be the following
    
    Final Viscous Resistance calculation: R_v = (8*mu/pi) * INT_0_L(max{gamma, zeta} * 1/(R(x)^4) dx)

    Units: Q = mL/s (cm^3/s), L = cm, R = cm, K = -, a = 1/cm, R_v = g/(s*cm^4)

    Created variable:
        - self.viscous_resistances: List of the viscous resistances calculated at every centerline point
    '''
    def calculate_viscous_resistances(self):
        self.viscous_resistances = [] #List for viscous resistances
        CONST_TERM = 8*self.dyn_viscosity/np.pi #The constant term in the viscous resistance equation

        #Resistance contribution of all the centerline points until and excluding the last point
        #The length (L) is half the distance from the last point to this point and half the distance from this point to the next
        #Not using the first or last point since their radius values are a little funky and they are in the flow extension region anyways
        for i in range(1, len(self.point_array_np)-1):
            #Calculate the distances between the points
            back_L = self.length_array[i] - self.length_array[i-1]
            forward_L = self.length_array[i+1] - self.length_array[i]
            #Calculate the length of the segment for this centerline point
            L_i = back_L/2 + forward_L/2

            #Getting the radius at this point
            rad = self.radius_array_np[i]

            #Calculating the unsteady term - zeta
            unsteady_term = rad * np.sqrt(self.density / self.dyn_viscosity)
            
            #Calculating the curvature term - gamma
            curv = self.curvature_array_np[i]
            K_i = self.reynolds * np.sqrt(rad / curv)
            curve_res_i = 0.1033 * np.sqrt(K_i) * ((1+(1.729 / K_i)) ** 0.5 - (1.315 - np.sqrt(K_i))) ** -3 #Multiplier to add the curvature resistance term
            
            #The viscous resistance "multiplier" is the maximum of gamma and zeta
            multiplier = max(curve_res_i, 1)

            #Calculate the viscous resistance at this centerline point
            visc_res = (CONST_TERM * L_i * multiplier) / (rad ** 4)
            self.viscous_resistances.append(visc_res)

    '''
    Creating the arrays for the local minimum and local maximum indices

    Method used:
        - argrelextrema: Scipy method finding the local minimum/maximum
            -order = 3: For each point, looks at the 3 points upstream and downstream to determine local maximum/minimum
    
    Return:
        - minima_indices: Numpy array containing the indices for each of the local minimum
        - maxima_indices: Numpy array containing the indices for each of the local maximum
        - start_min: Boolean - True if there is a local minimium before a local maximum, False if local max before local min
    '''
    def create_min_max_array(self):
        minima_indices = argrelextrema(self.radius_array_np, np.less, order=3)[0] #Gets the inidices of the local minima - order = 3 means that 3 points on each side used for comparison to reduce noise
        maxima_indices = argrelextrema(self.radius_array_np, np.greater, order=3)[0]
        #start_min = minima_indices[0] < maxima_indices[0] #True if the index of the first minima is less than the index of the first maxima
        
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
            A_0 = np.pi * self.radius_array_np[extrema_array[0]] ** 2
            A_s = np.pi * self.radius_array_np[extrema_array[1]] ** 2

            delta_R = self.calculate_added_resistance(A_s, A_0)
            exp_res_dict[min_indices[i]] = delta_R #C: i is undefined - i think it supposed to be 0
            expansion_resistance += delta_R

            #The first and last values aren't handled by the for loop
            for i in range(1, len(min_indices)-1):
                extrema_i = np.where(extrema_array == min_indices[i])[0][0]
                A_s = np.pi * self.radius_array_np[min_indices[i]] ** 2
                A_0 = np.pi * ((self.radius_array_np[extrema_array[extrema_i-1]] + self.radius_array_np[extrema_array[extrema_i+1]]) / 2) ** 2
                
                delta_R = self.calculate_added_resistance(A_s, A_0)
                exp_res_dict[min_indices[i]] = delta_R
                expansion_resistance += delta_R
            
            if last == "min":
                A_s = np.pi * self.radius_array_np[extrema_array[-1]] ** 2
                A_0 = np.pi * self.radius_array_np[extrema_array[-2]] ** 2
            else:
                A_s = np.pi * self.radius_array_np[extrema_array[-2]] ** 2
                A_0 = np.pi * ((self.radius_array_np[extrema_array[-3]] + self.radius_array_np[extrema_array[-1]]) / 2) ** 2
            
            delta_R = self.calculate_added_resistance(A_s, A_0)
            exp_res_dict[min_indices[i]] = delta_R
            expansion_resistance += delta_R
        
        #If the first element is a maxima, need to treat a little differently
        else:
            for i in range(0, len(min_indices)-1):
                extrema_i = np.where(extrema_array == min_indices[i])[0][0]
                A_s = np.pi * self.radius_array_np[min_indices[i]] ** 2
                A_0 = np.pi * ((self.radius_array_np[extrema_array[extrema_i-1]] + self.radius_array_np[extrema_array[extrema_i+1]]) / 2) ** 2
                
                delta_R = self.calculate_added_resistance(A_s, A_0)
                exp_res_dict[min_indices[i]] = delta_R
                expansion_resistance += delta_R

            if last == "min":
                A_s = np.pi * self.radius_array_np[extrema_array[-1]] ** 2
                A_0 = np.pi * self.radius_array_np[extrema_array[-2]] ** 2
            else:
                A_s = np.pi * self.radius_array_np[extrema_array[-2]] ** 2
                A_0 = np.pi * ((self.radius_array_np[extrema_array[-3]] + self.radius_array_np[extrema_array[-1]]) / 2) ** 2
            
            delta_R = self.calculate_added_resistance(A_s, A_0)
            exp_res_dict[min_indices[-1]] = delta_R
            expansion_resistance += delta_R

        self.expansion_resistances = expansion_resistance
        self.exp_res_dict = exp_res_dict

    '''
    Linearly adding the expansion resistance from the start of the expansion region (local minimum) 
    to the end (downstream local maximum)

    Parameters:
        - key: The id of the local minimum (index in lists)
        - val: Total expansion resistance to be applied over the expansion region
        - resistances: List of previously calculated resistance values for every point
        - max_indices: List of the local maximum indices

    Return:
        - resistances: List of calculated resistance values for every point after expansion resistance added
    '''
    def add_linear_expansion_resistance(self, key, val, resistances, max_indices):
        #Find the next maximum after this local minimum
        next_max = max_indices[max_indices > key]
        if len(next_max) == 0:
            #If there is no downstream maximum - apply entirely at the minimum point
            resistances[key] += val
            return resistances
        
        next_max_idx = next_max[0]

        #Points in the recovery region (inclusive of both endpoints)
        region_indices = list(range(key, next_max_idx + 1))

        #Equal share per point
        r_per_point = val / len(region_indices)
        for idx in region_indices:
            #viscous resistances is offset by 1 (starts at centerline point 1)
            res_idx = idx - 1
            if 0 <= res_idx < len(self.viscous_resistances):
                # self.viscous_resistances[res_idx] += r_per_point
                resistances[res_idx] += r_per_point
        
        return resistances

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
                    resistances[res_idx] += r_per_point
            
        else:
            for idx, w in zip(region_indices, weights):
                res_idx = idx - 1
                if 0 <= res_idx < len(self.viscous_resistances):
                    resistances[res_idx] += val * (w/total_weight)
        
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
                delta_a = np.pi * (self.radius_array_np[idx] ** 2) - np.pi * (self.radius_array_np[idx - 1] ** 2)
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
    Calculating the pressure drop at every point from the calculated resistance values
    '''
    def calculate_pressures(self):
        self.pressure_drops_mmHg = [] #List of pressure drops due to resistances of each segment
        self.pressures_mmHg = [] #List of pressures at each point

        #Calculating Total resistance
        resistances = self.viscous_resistances.copy() #Viscous resistance term
        if self.expansion == 2 or self.expansion == 3 or self.expansion == 4:
            _, max_indices = self.create_min_max_array()
        #Adding expansion resistance
        for key, val in self.exp_res_dict.items():
            if self.expansion == 1:
                resistances[key] += val
            elif self.expansion == 2:
                #Linear expansion resistance
                resistances = self.add_linear_expansion_resistance(key, val, resistances, max_indices) #This doesn't return anything that is added to resistances. This won't work.
            elif self.expansion == 3:
                resistances = self.add_proportional_expansion_resistance(key, val, resistances, max_indices)
            elif self.expansion == 4:
                resistances = self.add_proportional_to_area_expansion_resisance(key, val, resistances, max_indices)
            else:
                raise ValueError(f"EXPANSION flag must be set to a value between 0 and 4 inclusive. Not {self.expansion}")
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
        # visc_pressure = sum(self.viscous_resistances) * self.flow_rate / 1333.2
        # exp_pressure = sum(resistances) * self.flow_rate / 1333.2 - visc_pressure
        visc_pressure = 0
        exp_pressure = 0
        for i in range(len(resistances)):
            visc_res = self.viscous_resistances[i]
            exp_res = resistances[i] - self.viscous_resistances[i]

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
        
    def generate_pressure_drop_contributions_plots(self):
        # Create a figure with 1 row and 2 columns for side-by-side plots
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Using the same centerline slicing as your previous plots
        x = self.length_array[11:-11]
        
        # Slice the new pressure drop lists to match the length of x
        p_drop_visc = self.visc_pressures_mmHg[10:-10]
        p_drop_exp = self.exp_pressures_mmHg[10:-10]
        
        # 
        ax.plot(x, p_drop_visc, color='green', linewidth=2, label="Viscous Term")
        ax.plot(x, p_drop_exp, color='blue', linewidth=2, label="Expansion Term")

        ax.set_xlabel("Length Along Centerline (cm)", fontsize=16)
        ax.set_ylabel("Pressure Drop (mmHg)", fontsize=16)
        ax.set_title("Viscous vs Expansion Pressure Drop in LPM", fontsize=20)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(fontsize=16)
        
        plt.tight_layout()
        
        # Save and show the figure
        save_path = f"{self.figure_save_folder}/{self.debug_case_name}_pdrop_contributions_exp_{self.expansion}_curv_{self.curvature}.png"
        plt.savefig(save_path, dpi=300)
        plt.show()

    '''
    Generating the plots of the pressures along the centerline and the pressure drops along the centerline
    '''
    def generate_pressure_plots(self):
        fig, ax1 = plt.subplots(figsize=(10,6))
        # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7,12))
        x = self.length_array[11:-11]

        #Loading Case C's extracted data
        case_c_df = pd.read_csv("../extracted_pressure_data/Case_C_individual.csv")
        cfd_x_cm = case_c_df["dist"]
        cfd_pressure = case_c_df["pcen"]
        cfd_bernoulli = case_c_df["pber"]
        
        #Loading Rojin's data to compare against
        # 2. Loading Hemodynamics CFD Data
        # The CSV has a comment on the first row ("# Q_actual=5.5797 mL/s"), so we skip it.
        csv_filepath = "/home/kabir/masters_files/CFD_Results/From_Rojin/PTSeg028_base_0p64_centerline_hemodynamics_Qin5.58mLs.csv"
        hemo_df = pd.read_csv(csv_filepath, skiprows=1)
        
        # Clean up column names (removes the '# ' and any trailing spaces)
        hemo_df.columns = [col.replace('#', '').strip() for col in hemo_df.columns]
        
        # Extract distance (mm) and pressure (mmHg)
        cfd_x_mm = hemo_df['cl_dist_mm'] / 10.0 #THIS IS ACC CM
        cfd_pressure_rojin = hemo_df['pressure_mmHg'] - hemo_df['pressure_mmHg'][1312]

        # 3. Numerical Comparison (Interpolation)
        # To compare point-by-point, interpolate the CFD pressure at your ROM x-coordinates.
        # Note: numpy.interp requires the x-array to be strictly increasing, so we sort it first.
        sort_idx = np.argsort(cfd_x_mm.values)
        cfd_x_sorted = cfd_x_mm.values[sort_idx]
        cfd_p_sorted = cfd_pressure_rojin.values[sort_idx]
        
        cfd_interp = np.interp(x, cfd_x_sorted, cfd_p_sorted)
        mean_abs_error = np.mean(np.abs(self.pressure_drops_mmHg[10:-10] - cfd_interp))

        # Pressure along vessel
        #ax1.scatter(x, self.pressures_mmHg[10:-10], c=colours, s=20)
        ax1.plot(x, self.pressures_mmHg[10:-10] , color='red', linewidth=2, label="LPM")
        #ax1.plot(cfd_x_cm, cfd_pressure, color="black", linewidth=1.5, label='3D CFD')
        #ax1.plot(cfd_x_cm, cfd_bernoulli, color="blue", linewidth=1.5, linestyle="--", label="Bernoulli Data")
        # Line for CFD Hemodynamics Data
        ax1.plot(cfd_x_mm, cfd_pressure_rojin, color="black", linewidth=2, label='3D CFD')
        ax1.set_xlabel("Length Along Centerline (cm)", fontsize=16)
        ax1.set_ylabel("Pressure (mmHg)", fontsize=16)
        ax1.set_title("Pressure Along Segment", fontsize=20)
        ax1.grid(True)
        ax1.legend(fontsize=16)
        print(self.pressures_mmHg[-10])
        print((cfd_pressure_rojin[1]))

        # Pressure drops at each segment
        # ax2.bar(x, self.pressure_drops_mmHg[10:-10], color="black", alpha=0.3)
        # ax2.set_xlabel("Length Along Centerline (ccm)")
        # ax2.set_ylabel("Pressure Drop (mmHg)")
        # ax2.set_title("Pressure Drop at Each Point Along Vessel")
        # ax2.grid(True, axis='y')

        plt.tight_layout()
        plt.savefig(f"{self.figure_save_folder}/{self.debug_case_name}_pressure_results_exp_{self.expansion}_curv_{self.curvature}.png", dpi=300)
        plt.show()

    def compare_distances(self, csv_path):
        # 1. Load CSV and clean columns
        hemo_df = pd.read_csv(csv_path, skiprows=1)
        hemo_df.columns = [col.replace('#', '').strip() for col in hemo_df.columns]
        
        # 2. Convert CSV distance to cm to match self.length_array units
        csv_dist_cm = hemo_df['cl_dist_mm'].values / 10.0
        rom_dist_cm = self.length_array[::-1]
        
        # 3. Check for length mismatch
        if len(csv_dist_cm) != len(rom_dist_cm):
            print(f"Warning: Length mismatch! CSV: {len(csv_dist_cm)}, ROM: {len(rom_dist_cm)}")
            # If they differ, you must use interpolation instead of element-wise subtraction
            return None

        # 4. Element-wise comparison
        # Note: If self.length_array is [0...max] and CSV is [max...0], 
        # you may need to reverse one: csv_dist_cm = csv_dist_cm[::-1]
        diff = rom_dist_cm - csv_dist_cm
        
        # 5. Statistics
        print(f"Mean Difference: {np.mean(np.abs(diff)):.4f} cm")
        print(f"Max Difference: {np.max(np.abs(diff)):.4f} cm")
        
        # fig, ax = plt.subplots(figsize=(12,6))


    def debug(self, txt_file_name, desc):
        text_lines = []
        text_lines.append(f"\nDescription: {desc}\n")
        if hasattr(self, "viscous_resistances"):
            v_res_sum = sum(self.viscous_resistances)/1333.2
            text_lines.append(f"Viscous Total Resistance (+Curvature if CURVATURE = 1): {v_res_sum}\n")
            text_lines.append(f"Pressure drop due to viscous losses: {v_res_sum * self.flow_rate}\n")
        if hasattr(self, "expansion_resistances"):
            text_lines.append(f"Expansion Total Resistance: {self.expansion_resistances/1333.2}\n")
            text_lines.append(f"Pressure drop due to expansion losses: {self.expansion_resistances * self.flow_rate / 1333.2}\n")

        text_lines.append('\n')

        #Actually writing to the text file
        folder = Path(txt_file_name).parent
        folder.mkdir(parents=True, exist_ok=True) #Creates the folder if it doesn't exist already
        with open(txt_file_name, "a") as f:
            f.writelines(text_lines)

    '''
    Function to run everything in the correct order based on the parameter given during class initialization,
    to make this class easy to use.
    '''
    def run(self):
        #C: ADD CHECKS TO ENSURE THAT THE PARAMETERS GIVEN ARE VALID

        #Calculate viscous resistance
        self.calculate_viscous_resistances()

        #Calculate expansion resistance if necessary
        if self.expansion == 0:
            self.calculate_pressures_no_exp()
        else:
            self.calculate_expansion_resistances()
            self.calculate_pressures()

        #Output to debug file if desired
        if self.debug:
            desc = f"{'='*50}\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\t{self.debug_case_name}->\tEXPANSION = {self.expansion}\tCURVATURE = {self.curvature}"
            self.debug(txt_file_name=self.debug_file_path, desc=desc)
        
        # lp.compare_distances("/home/kabir/masters_files/CFD_Results/From_Rojin/PTSeg028_base_0p64_centerline_hemodynamics_Qin5.58mLs.csv")
        lp.generate_pressure_plots()
        lp.generate_pressure_drop_contributions_plots()
        #lp.compare_hemodynamics_pressure("/home/kabir/masters_files/CFD_Results/From_Rojin/PTSeg028_base_0p64_centerline_hemodynamics_Qin5.58mLs.csv")

def main():
    #CONSTANTS
    BLOOD_DYNAMIC_VISCOSITY = 0.04 #dynamic viscosity mu value [Poise]
    INLET_FLOW_RATE = 5.58 #mL/s - same as Back to Bernoulli paper
    KT = 1.52 #Same as Mirramezani paper
    DENSITY = 1.06 #g/mL or g/cm^3
    REYNOLDS_NUMBER = 300 #Reynold's number for cerebral venous system - 300 is a placeholder value for now
        
    EXPANSION = 3 #0 (no expansion), 1(exp res all at one point), 2(exp res applied linearly), 3(exp res applied proportional to radius), 4(exp res applied proportional to area)
    CURVATURE = 1 #0 - no curvature resistance term added, 1 - curvature resistance term added
    
    try:
        import config
        CLINE_FILE_PATH = config.CLINE_FILE_PATH
        FIGURE_SAVE_FOLDER = config.FIGURE_SAVE_FOLDER #Path to folder where the figures should be saved
        DEBUG = True #Flag to output verbose results to debug text file specified below
        if DEBUG:
            DEBUG_FILE_PATH = config.DEBUG_FILE_PATH #File path for debug info file
            DEBUG_CASE_NAME = config.DEBUG_CASE_NAME #A name to identify in the debug file
            debug_options = [DEBUG_FILE_PATH, DEBUG_CASE_NAME]
        else:
            debug_options = None
    except Exception as e:
        raise Exception(f"Please ensure that the config.py file is present in the same folder as this file and all the necessary variables are present:\n{e}")

    lp = LumpedParameter(
        cline_file=CLINE_FILE_PATH,
        Q=INLET_FLOW_RATE,
        rho=DENSITY,
        Kt=KT,
        mu=BLOOD_DYNAMIC_VISCOSITY,
        re=REYNOLDS_NUMBER,
        curv=CURVATURE,
        exp=EXPANSION,
        fig_save_folder=FIGURE_SAVE_FOLDER,
        debug_options=debug_options
    )

    lp.run()

if __name__ == "__main__":
    main()