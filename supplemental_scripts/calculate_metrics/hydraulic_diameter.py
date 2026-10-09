'''
Calculates metrics (not just hydraulic diameter)

Inputs (from config.py file):
- 
-
- 
- 

Outputs (created files):
- if CREATE_EXCEL:
    - Excel (.xlsx) file containing just point ID, CSA, and Perimeter
- if CREATE_CSV:
    - CSV (.csv) file containing all created data
- if CREATE_VTP:
    - VTP (.vtp) file containing all the same data that was in the given centerline file but adding:
        - Cross Sectional Area
        - Perimeter
        - Hydraulic Diameter
        - DH:MISR Ratio (MISR = Maximum Inscribed Sphere Radius)
'''

import numpy as np 
import vtk
from vtk.util.numpy_support import vtk_to_numpy
vtk.vtkObject.GlobalWarningDisplayOff()
import csv
import pandas as pd
import pyvista as pv
import numpy as np
import os
from pathlib import Path

'''
Loading the centerline and returning arrays for quantities that we need

Parameter:
    vtp_path: Given path to the centerline file

Returns:
    points: Array of the points
    tangents: numpy array of the tangents for each point
    misr: numpy array of the maximally inscribed sphere radius for each point
    curvs: numpy array of the curvature values for each point
    num_points: Integer of total number of centerline points for this case
'''
def load_centerline(vtp_path: str):
    #Using pyvista to load the centerline natively
    polydata = pv.read(vtp_path)
    num_points = polydata.n_points
    points = polydata.points

    if "FrenetTangent" not in polydata.point_data:
        raise ValueError("No FrenetTangent in VTP centerline file - run Geometry Tools first and retry")
    tangents = np.array(polydata.point_data["FrenetTangent"])

    if "MaximumInscribedSphereRadius" not in polydata.point_data:
        raise ValueError("No MaximumInscribedSphereRadius in VTP centerline file - run Geometry Tools first and retry")
    misr = np.array(polydata.point_data["MaximumInscribedSphereRadius"])

    if "Curvature" not in polydata.point_data:
        raise ValueError("No Curvature in VTP centerline file - run Geometry Tools first and retry")
    curvs = np.array(polydata.point_data["Curvature"])
    
    return points, tangents, misr, curvs, num_points

def cross_section_metrics(mesh: pv.PolyData, origin:np.ndarray, normal:np.ndarray):
    normal = normal / (np.linalg.norm(normal) + 1e-15)

    #Slicing the mesh with the already calculated normal plane
    try:
        slc = mesh.slice(normal=normal, origin=origin)
    except:
        return None, None, None, None, None

    if slc.n_points == 0:
        #2D tangent plane does not intersect any triangles in MESH file
        return None, None, None, None, None
    
    #Extracting only the region closest to the centerline origin
    #This avoids adding up areas from neighbouring vessel branches that the infinite plane might cut
    conn = vtk.vtkPolyDataConnectivityFilter()
    conn.SetInputData(slc)
    conn.SetExtractionModeToClosestPointRegion()
    conn.SetClosestPoint(origin[0], origin[1], origin[2])
    conn.Update()
    closest_slc = pv.wrap(conn.GetOutput())

    if closest_slc.n_points == 0:
        return None, None, None, None, None

    #Calculate perimeter using cell sizes (sum of line segment lengths)
    slc_sized = closest_slc.compute_cell_sizes(length=True, area=False, volume=False)
    perimeter = np.sum(slc_sized.cell_data["Length"])

    #Triangulate the 3D contour to calculate internal area and generate a solid 2D planar surface
    triangulator = vtk.vtkContourTriangulator()
    triangulator.SetInputData(closest_slc)
    triangulator.Update()

    solid_slice = pv.wrap(triangulator.GetOutput())
    if solid_slice.n_points == 0:
        return None, None, None, None

    area = solid_slice.area
    Dh = 4.0 * area / perimeter if perimeter > 1e-12 else 0.0
    if Dh == None: 
        print(f'Dh is None. Other values: Area - {area}, perimeter - {perimeter} for point at origin {origin}')
    return area, perimeter, Dh, closest_slc, solid_slice

def write_output_vtp(VTP, c_areas, c_perimeters, c_dhs, c_ratios):
    #Loading centerline again to attach new data
    out_centerline = pv.read(VTP)
    #Adding new arrays to point data
    out_centerline.point_data["CrossSectionArea"] = c_areas
    out_centerline.point_data["CrossSectionPerimeter"] = c_perimeters
    out_centerline.point_data["HydraulicDiameter"] = c_dhs
    out_centerline.point_data["DhMISRRatio"] = c_ratios

    OUT_VTP = VTP.replace(".vtp", "_with_metrics.vtp") #Creating file path for new vtp centerline file with metrics
    out_centerline.save(OUT_VTP)
    print(f"Saved updated centerline file with area, perimeter, and hydraulic diameter added to point data to: {OUT_VTP}")

def create_excel_file(vtp_file_path, output_dir, c_areas, c_perimeters):
    data = {
        "Area": c_areas,
        "Perimeter": c_perimeters
    }

    df = pd.DataFrame(data)

    df.to_excel(output_dir, index=True) #Writing the DataFrame to the Excel file

def write_csv(CSV, results):
    fieldnames = list(results[0].keys())
    with open(CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

def validate_csv(csv_path: str, vtp_path: str):
    print("Validating created CSV file")
    df = pd.read_csv(csv_path)
    mesh = pv.read(vtp_path)

    vtp_points = mesh.points
    vtp_data = mesh.point_data

    mapping = {
        "MISR": "MaximumInscribedSphereRadius",
        "tangent_x": "FrenetTangent",
        "tangent_y": "FrenetTangent",
        "tangent_z": "FrenetTangent"
    }

    csv_points = df[['x', 'y', 'z']].values
    print()
    if len(df) != len(vtp_points):
        print(f"Warning: Row count mismatch!\nCSV: {len(csv_points)}, VTP: {len(vtp_points)}")
        limit = min(len(csv_points), len(vtp_points))
        csv_points = csv_points[:limit]
        vtp_points = vtp_points[:limit]
    
    coord_diff = np.linalg.norm(csv_points - vtp_points, axis=1)
    max_coord_err = np.max(coord_diff)
    print(f"Max Coordinate Difference | {max_coord_err:.6f}")

    scalars_to_check = ['MISR']
    
    for col in scalars_to_check:
        vtp_name = mapping.get(col, col)

        if vtp_name in vtp_data:
            diff = np.abs(df[col].values - vtp_data[vtp_name])
            print(f"{col:<20} | Max Diff: {np.max(diff):.2e}")
            is_consistent = np.allclose(df['MISR'].values, vtp_data['MaximumInscribedSphereRadius'], atol=1e-12)
            print(f"Are all values within machine tolerance: {is_consistent}")
        else:
            print(f"{col:<20} | Array '{vtp_name}' not found in VTP")
    
    if 'FrenetTangent' in vtp_data:
        vtp_tangents = vtp_data['FrenetTangent']
        csv_tangents = df[['tangent_x', 'tangent_y', 'tangent_z']].values
        tangent_diff = np.abs(csv_tangents - vtp_tangents)
        print(f"{'Tangents (Combined)':<20} | Max Diff: {np.max(tangent_diff):2e}")

def main():
    try:
        from . import config
        MESH = config.hd_mesh
        VTP = config.hd_vtp
        CSV = config.hd_csv
        VIZ_DIR = config.hd_viz_dir
        EXCEL_OUTPUT = config.hd_excel_output_dir
        if not os.path.exists(MESH):
            raise FileNotFoundError("Could not find surface mesh file. Please enter valid path")
        if not os.path.exists(VTP):
            raise FileNotFoundError("Could not find VTP centerline file. Please enter valid path")
    except Exception as e:
        raise Exception(f"Ran into an error getting file path from config file: {e}")

    SKIP = 1
    NO_PROGRESS = False
    SAVE_OUTLINES = False
    CREATE_CSV = True
    CREATE_EXCEL = False

    if SAVE_OUTLINES:
        print(f"Creating output directory (if it doesn't exist)")
        os.makedirs(VIZ_DIR, exist_ok=True)

    print(f"Loading mesh: {MESH}...")
    mesh = pv.read(MESH)

    #Pyvista watertightness check
    edges = mesh.extract_feature_edges(boundary_edges=True, non_manifold_edges=True, feature_edges=False, manifold_edges=False)
    is_watertight = (edges.n_points == 0)

    print(f"  Vertices: {mesh.n_points:,}\tFaces: {mesh.n_cells:,}\tWatertight: {is_watertight}")
    if not is_watertight and MESH.endswith(".stl") or MESH.endswith(".vtu"):
        raise ValueError("The mesh is not watertight (has open edges/faces). Please fix and re-run")
    
    print(f"Loading centerline: {VTP}...")
    points, tangents, misr, curvs, n_total = load_centerline(VTP)
    print(f"Centerline points: {n_total:,}")

    indices = range(0, n_total, SKIP)
    results = []
    skipped = 0
    skipped_pts = []

    #Pre-allocating arrays for the centerline VTP file
    c_areas = np.full(n_total, np.nan)
    c_perimeters = np.full(n_total, np.nan)
    c_dhs = np.full(n_total, np.nan)
    c_ratios = np.full(n_total, np.nan)

    for count, i in enumerate(indices):
        #Updating print statement with progress
        if not NO_PROGRESS and count % max(1, len(indices)//20) == 0:
            pct = 100 * count / max(1, len(indices))
            print(f'Progress:{pct:5.1f}% point {i}/{n_total}', end="\r", flush=True) #Updating the progress tracker every 5%

        area, perimeter, Dh, outline, solid_slice = cross_section_metrics(mesh, points[i], tangents[i])

        if area is None:
            #If area was returned as None, it means there was an error so we are skipping the point
            skipped += 1
            skipped_pts.append(count)
            area = perimeter = Dh = float("nan")
        else:
            Dh_misr_ratio = Dh / (2*misr[i]) if misr[i] != 0 else 0.0
            #Saving the output outline, planes, and centerline point files
            if SAVE_OUTLINES:
                outline.save(os.path.join(VIZ_DIR, f"outline_{i:04d}.vtp"))
                solid_slice.save(os.path.join(VIZ_DIR, f"plane_{i:04d}.vtp"))
                cline_pt = pv.PolyData(points[i])
                cline_pt.save(os.path.join(VIZ_DIR, f"centerline_point_{i:04d}.vtp"))

            #Storing calculated metrices into arrays
            c_areas[i] = area
            c_perimeters[i] = perimeter
            c_dhs[i] = Dh
            c_ratios[i] = Dh_misr_ratio

        if CREATE_CSV:
            #Appending to the results array for putting into the csv file
            results.append({
                "point_index": i,
                "x": points[i, 0],
                "y": points[i, 1],
                "z": points[i, 2],
                "tangent_x": tangents[i, 0],
                "tangent_y": tangents[i, 1],
                "tangent_z": tangents[i, 2],
                "MISR": misr[i],
                "area": area,
                "perimeter": perimeter,
                "hydraulic_diameter": Dh,
                "Dh_2*MISR_Ratio": Dh_misr_ratio
            })

    if CREATE_CSV:
        write_csv(CSV, results)

    write_output_vtp(VTP, c_areas, c_perimeters, c_dhs, c_ratios)

    #Validation metrics
    valid = sum(1 for r in results if not np.isnan(r["area"]))
    print()
    print(f"\nDone. {valid}/{len(results)} points had valid cross-sections")
    print(f"Skipped (no intersection between plane and surface mesh triangles): {skipped}")
    print(f"Points skipped: {skipped_pts}")
    if CREATE_CSV:
        validate_csv(csv_path=CSV, vtp_path=VTP)
        print(f"CSV Results written to: {CSV}")
    if SAVE_OUTLINES:
        print(f"Visualizations saved to folder: '{VIZ_DIR}/' (Load into Paraview to scroll)")

    areas = np.array([r["area"] for r in results if not np.isnan(r["area"])])
    Dhs = np.array([r["hydraulic_diameter"] for r in results if not np.isnan(r["hydraulic_diameter"])])
    if len(areas):
        print(f"Cross-sectional area - min: {areas.min():.3f}")
        print(f"                       mean: {areas.mean():.3f}")
        print(f"                       max: {areas.max():.3f}")
        print(f"Hydraulic diameter - min: {Dhs.min():.3f}")
        print(f"                     mean: {Dhs.mean():.3f}")
        print(f"                     max: {Dhs.max():.3f}")

    if CREATE_EXCEL:
        create_excel_file(VTP, Path(EXCEL_OUTPUT), c_areas, c_perimeters)

if __name__ == "__main__":
    main()