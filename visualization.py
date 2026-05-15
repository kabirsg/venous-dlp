import pyvista as pv
import numpy as np
from scipy.spatial import cKDTree

"""
Script to map 1D centerline metrics (Area, Perimeter, Dh) 
onto a 3D surface mesh using nearest-neighbor mapping.
"""
def main():
    # 1. Define your file paths
    try:
        import config
        SURFACE_STL = config.v_stl
        CENTERLINE_VTP = config.v_cline_vtp_with_metrics
        OUT_SURFACE_VTP = config.v_out_surface_vtp
    except:
        raise FileNotFoundError("Could not find config file. Please ensure it's in the same directory as this script and rerun.")

    # 2. Load the data
    print(f"Loading 3D Surface: {SURFACE_STL}")
    surf = pv.read(SURFACE_STL)
    
    print(f"Loading Centerline: {CENTERLINE_VTP}")
    cl = pv.read(CENTERLINE_VTP)

    # Verify that the metrics exist on the centerline
    if "HydraulicDiameter" not in cl.point_data:
        raise ValueError("The centerline VTP is missing the 'HydraulicDiameter' array. "
                         "Make sure you are loading the '_with_metrics.vtp' file.")

    # 3. Extract coordinates
    surf_pts = np.array(surf.points)
    cl_pts = np.array(cl.points)

    print(f"\nMapping {cl.n_points:,} centerline points onto {surf.n_points:,} surface vertices...")
    
    # 4. Build a KD-Tree for nearest-neighbor lookups
    tree = cKDTree(cl_pts)
    
    # For every surface point, find the index of the closest centerline point
    _, nearest_cl_indices = tree.query(surf_pts)

    # 5. Copy the data arrays over
    metrics_to_map = ["CrossSectionArea", "CrossSectionPerimeter", "HydraulicDiameter", "MaximumInscribedSphereRadius"]
    
    for metric in metrics_to_map:
        if metric in cl.point_data:
            # Extract the 1D array from the centerline
            cl_array = np.array(cl.point_data[metric])
            
            # Create a new array for the surface by matching indices
            surf_array = cl_array[nearest_cl_indices]
            
            # Attach the new array to the 3D surface
            surf.point_data[metric] = surf_array
            print(f"  -> Successfully mapped: {metric}")

    # 6. Save the new surface
    print(f"\nSaving mapped 3D surface to: {OUT_SURFACE_VTP}")
    surf.save(OUT_SURFACE_VTP)
    print("Done! You can now load this .vtp file into Paraview and color it by these metrics.")

if __name__ == "__main__":
    main()