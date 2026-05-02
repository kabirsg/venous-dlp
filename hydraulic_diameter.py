import argparse
import sys
import numpy as np
import vtk 
import trimesh 
import csv
import pandas as pd
import pyvista as pv

"""
Script to calculate the hydraulic diameter, given a centerline and an STL file

THIS REQUIRES A VTK VERSION < 9.3.0. WILL THROW AN ERROR WITH vtkExtractEdges SINCE THIS WAS REMOVED FROM 9.3.0 BUT PYVISTA NEEDS IT
"""

def load_centerline(vtp_path: str):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(vtp_path)
    reader.Update()
    polydata = reader.GetOutput()
    num_points = polydata.GetNumberOfPoints()

    points = np.array([polydata.GetPoint(i) for i in range(num_points)], dtype=np.float64)
    point_data = polydata.GetPointData()

    tangent_arr = point_data.GetArray("FrenetTangent")
    if tangent_arr is None:
        raise ValueError("VTP file has no 'FrenetTangent' point array. Run vmtkCenterlineGeometry (or Geometry Tools) first.")
    tangents = np.array([tangent_arr.GetTuple3(i) for i in range(num_points)], dtype=np.float64)
    
    misr_arr = point_data.GetArray("MaximumInscribedSphereRadius")
    misr = (
        np.array([misr_arr.GetValue(i) for i in range(num_points)], dtype=np.float64)
        if misr_arr is not None
        else np.full(num_points, np.nan)
    )

    return points, tangents, misr, num_points

def cross_section_metrics(mesh: trimesh.Trimesh, origin: np.ndarray, normal: np.ndarray):
    normal = normal / (np.linalg.norm(normal) + 1e-15)

    #mesh.section returns a Path3D or None
    section_3d = mesh.section(plane_origin=origin, plane_normal=normal)
    if section_3d is None:
        return None, None, None #Failed
    
    #Project onto a 2-D plane for area/perimeter calculation
    try:
        section_2d, _ = section_3d.to_2D()
    except:
        return None, None, None #Failed
    
    area = section_2d.area
    perimeter = section_2d.length
    Dh = 4.0 * area / perimeter if perimeter > 1e-12 else 0.0

    return area, perimeter, Dh

'''
Compares what's in the CSV is the same as what's in the VTP
'''
def validate_csv(csv_path: str, vtp_path: str):
    #Loading the CSV data
    df = pd.read_csv(csv_path)
    #Loading VTP data
    mesh = pv.read(vtp_path)

    #Extracting coordinates from VTP
    #mesh.points is an Nx3 array (x, y, z)
    vtp_points= mesh.points
    vtp_data = mesh.point_data

    #Define Column Mapping
    mapping = {
        'MISR': 'MaximumInscribedSphereRadius',
        'tangent_x': 'FrenetTangent',
        'tangent_y': 'FrenetTangent',
        'tangent_z': 'FrenetTangent'        
    }

    csv_points = df[['x', 'y', 'z']].values
    print()
    #Check if the dimensions match
    if len(df) != len(vtp_points):
        print(f"Warning: Row count mismatch!\nCSV: {len(csv_points)}, VTP: {len(vtp_points)}")
        #We truncate to the shortest for now
        limit = min(len(csv_points), len(vtp_points))
        csv_points = csv_points[:limit]
        vtp_points = vtp_points[:limit]
    
    coord_diff = np.linalg.norm(csv_points - vtp_points, axis=1)
    max_coord_err = np.max(coord_diff)
    print(f"Max Coordinate Difference   | {max_coord_err:.6f}")

    #Comparing scalar/vector fields
    scalars_to_check = ['MISR']

    for col in scalars_to_check:
        vtp_name = mapping.get(col, col) #Fallback to same name if not in mapping

        if vtp_name in vtp_data:
            diff = np.abs(df[col].values - vtp_data[vtp_name])
            print(f"{col:<20} | Max Diff: {np.max(diff):.2e}")
            # This will return True if the difference is within machine tolerance
            is_consistent = np.allclose(df['MISR'].values, vtp_data['MaximumInscribedSphereRadius'], atol=1e-12)
            print(f"Are all values within machine tolerance: {is_consistent}")
        else:
            print(f"{col:<20} | Array '{vtp_name}' not found in VTP")

    #Special Case: Tangents (Vector Component Comparison)
    if 'FrenetTangent' in vtp_data:
        vtp_tangents = vtp_data['FrenetTangent']
        csv_tangents = df[['tangent_x', 'tangent_y', 'tangent_z']].values
        tangent_diff = np.abs(csv_tangents - vtp_tangents)
        print(f"{'Tangents (Combined)':<20} | Max Diff: {np.max(tangent_diff):2e}")


def main():
    STL = "/home/kabir/masters_files/PT/PTSeg028_v3/PTSeg028_base_0p64.stl"
    VTP = "/home/kabir/masters_files/PT/PTSeg028_v3/PTSeg028_cl_centerline_graph_vmtk.vtp"
    CSV = "cross_sections.csv"
    SKIP = 1
    NO_PROGRESS = False

    #Checking mesh quality
    print(f"Loading mesh:        {STL}")
    mesh = trimesh.load_mesh(STL) #Loading the mesh in the STL file - Returns a trimesh.Trimesh object if the mesh is found
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("Could not load STL as a single Trimesh object.")
    print(f"  Vertices: {len(mesh.vertices):,}\tFaces: {len(mesh.faces):,}\tWatertight: {mesh.is_watertight}")
    
    if not mesh.is_watertight:
        raise ValueError("The STL Mesh is not watertight - this will cause errors. Please plug holes before running.")

    print(f"Loading centerline:  {VTP}")
    points, tangents, misr, n_total = load_centerline(VTP)
    print(f"  Centerline points: {n_total:,}")

    indices = range(0, n_total, SKIP)
    results = []
    skipped = 0

    for count, i in enumerate(indices):
        #Printing the progress bar onto the console
        if not NO_PROGRESS and count % max(1, len(indices) // 20) == 0:
            pct = 100 * count / max(1, len(indices))
            print(f"  {pct:5.1f}%  point {i}/{n_total}", end="\r", flush=True)

        area, perimeter, Dh = cross_section_metrics(mesh, points[i], tangents[i])

        if area is None:
            skipped += 1
            area = perimeter = Dh = float("nan")

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
        })

    print()  # newline after progress

    # ── write CSV ─────────────────────────────────────────────────────────────
    fieldnames = list(results[0].keys())
    with open(CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    valid = sum(1 for r in results if not np.isnan(r["area"]))
    print(f"\nDone. {valid}/{len(results)} points had valid cross-sections.")
    print(f"Skipped (no intersection): {skipped}")
    print(f"Results written to: {CSV}")

    # ── quick summary stats ───────────────────────────────────────────────────
    areas = np.array([r["area"] for r in results if not np.isnan(r["area"])])
    Dhs   = np.array([r["hydraulic_diameter"] for r in results if not np.isnan(r["hydraulic_diameter"])])
    if len(areas):
        print(f"\nCross-sectional area  —  min: {areas.min():.3f}  "
              f"mean: {areas.mean():.3f}  max: {areas.max():.3f}")
        print(f"Hydraulic diameter    —  min: {Dhs.min():.3f}  "
              f"mean: {Dhs.mean():.3f}  max: {Dhs.max():.3f}")
        
    #Quick check
    validate_csv(csv_path=CSV, vtp_path=VTP)


if __name__ == "__main__":
    main()