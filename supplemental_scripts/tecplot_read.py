import numpy as np
import re
import pandas as pd
import pickle

def read_tec(filepath):
    """
    Parse a Tecplot ASCII .tec file with multiple zones.
    Returns:
        variables: list of variable names
        zones: dict mapping zone title -> numpy array (n_points x n_vars)
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    variables = []
    zones = {}
    current_zone_name = None
    current_data = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Variables line
        if line.upper().startswith('VARIABLES'):
            # Everything after the '=' sign, comma-separated
            var_str = line.split('=', 1)[1]
            variables = [v.strip().strip('"') for v in var_str.split(',')]
            continue

        # Zone header
        if line.upper().startswith('ZONE'):
            # Save previous zone's data before starting a new one
            if current_zone_name is not None:
                zones[current_zone_name] = np.array(current_data, dtype=float)

            # Extract the zone title, e.g. ZONE T="Case A individual"
            match = re.search(r'T\s*=\s*"([^"]*)"', line)
            current_zone_name = match.group(1) if match else f"Zone_{len(zones)+1}"
            current_data = []
            continue

        # Otherwise, treat as a data line
        parts = line.split()
        try:
            row = [float(p) for p in parts]
            current_data.append(row)
        except ValueError:
            # Skip any non-numeric/unexpected lines (e.g. extra headers)
            continue

    # Save the last zone
    if current_zone_name is not None:
        zones[current_zone_name] = np.array(current_data, dtype=float)

    return variables, zones


if __name__ == "__main__":
    tecplot_file_loc = "../Gurnish_Data/pressures_feb26.tec"
    pickle_file_save_loc = "../Gurnish_Data/Gurnish_Case_Data.pkl"
    variables, zones = read_tec(tecplot_file_loc)

    zones_df = {name: pd.DataFrame(data, columns=variables) for name, data in zones.items()}
    
    with open(pickle_file_save_loc, "wb") as f:
        pickle.dump(zones_df, f)

    '''
    Later when you want to open this file and access the data inside:
    with open("Gurnish_Data/Gurnish_Case_Data.pkl", "rb") as f:
        zones_df = pickle.load(f)

    case_A_centerline_pressures = zones_df["Case A individual"]["pcen"]
    '''
