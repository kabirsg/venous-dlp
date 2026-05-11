import argparse 
import sys 
import numpy as np 
import vtk
vtk.vtkObject.GlobalWarningDisplayOff()
import csv
import pandas as pd
import pyvista as pv
import numpy as np
import os

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
    
    return points, tangents, misr, num_points

def cross_section_metrics(mesh: pv.PolyData, origin:np.ndarray, normal:np.ndarray):
    normal = normal / (np.linalg.norm(normal) + 1e-15)

    #Slicing the mesh with the already calculated normal plane
    try:
        slc = mesh.slice(normal=normal, origin=origin)
    except:
        return None, None, None, None

    if slc.n_points == 0:
        return None, None, None, None
    
    #Extracting only the region closest to the centerline origin
    #This avoids adding up areas from neighbouring vessel branches that the infinite plane might cut
    conn = vtk.vtkPolyDataConnectivityFilter()
    conn.SetInputData(slc)
    conn.SetExtractionModeToClosestPointRegion()
    conn.SetClosestPoint(origin[0], origin[1], origin[2])
    conn.Update()
    closest_slc = pv.wrap(conn.GetOutput())

    if closest_slc.n_points == 0:
        return None, None, None, None

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

    return area, perimeter, Dh, closest_slc, solid_slice

def validate_csv(csv_path: str, vtp_path: str):
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
        import config
        STL = config.hdp_stl
        VTP = config.hdp_vtp
        CSV = config.hdp_csv
        VIZ_DIR = config.hdp_viz_dir
    except:
        STL = ""
        VTP = ""
        CSV = "cross_sections.csv"
        VIZ_DIR = "cross_sections_viz"

    SKIP = 1
    NO_PROGRESS = False

    print(f"Creating output directory (if it doesn't exist)")
    os.makedirs(VIZ_DIR, exist_ok=True)

    print(f"Loading mesh: {STL}...")
    mesh = pv.read(STL)

    #Pyvista watertightness check
    edges = mesh.extract_feature_edges(boundary_edges=True, non_manifold_edges=True, feature_edges=False, manifold_edges=False)
    is_watertight = (edges.n_points == 0)

    print(f"  Vertices: {mesh.n_points:,}\tFaces: {mesh.n_cells:,}\tWatertight: {is_watertight}")
    if not is_watertight:
        raise ValueError("The STL mesh is not watertight (has open edges/faces). Please fix and re-run")
    
    print(f"Loading centerline: {VTP}...")
    points, tangents, misr, n_total = load_centerline(VTP)
    print(f"  Centerline points: {n_total:,}")

    indices = range(0, n_total, SKIP)
    results = []
    skipped = 0

    for count, i in enumerate(indices):
        if not NO_PROGRESS and count % max(1, len(indices)//20) == 0:
            pct = 100 * count / max(1, len(indices))
            print(f'{pct:5.1f}% point {i}/{n_total}', end="\r", flush=True)

        area, perimeter, Dh, outline, solid_slice = cross_section_metrics(mesh, points[i], tangents[i])

        if area is None:
            skipped += 1
            area = perimeter = Dh = float("nan")
        else:
            outline.save(os.path.join(VIZ_DIR, f"outline_{i:04d}.vtp"))
            solid_slice.save(os.path.join(VIZ_DIR, f"plane_{i:04d}.vtp"))

            cline_pt = pv.PolyData(points[i])
            cline_pt.save(os.path.join(VIZ_DIR, f"centerline_point_{i:04d}.vtp"))
        
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
            "hydraulic_diameter": Dh
        })

    print()

    #Write CSV
    fieldnames = list(results[0].keys())
    with open(CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    valid = sum(1 for r in results if not np.isnan(r["area"]))
    print(f"\nDone. {valid}/{len(results)} points had valid cross-sections")
    print(f"Skipped (no intersection): {skipped}")
    print(f"Results written to: {CSV}")
    print(f"Visualizations saved to folder: '{VIZ_DIR}/' (Load into Paraview to scroll)")

    areas = np.array([r["area"] for r in results if not np.isnan(r["area"])])
    Dhs = np.array([r["hydraulic_diameter"] for r in results if not np.isnan(r["hydraulic_diameter"])])
    if len(areas):
        print(f"\nCross-sectional area - min: {areas.min():.3f}")
        print(f"                       mean: {areas.mean():.3f}")
        print(f"                       max: {areas.max():.3f}")
        print(f"Hydraulic diameter - min: {Dhs.min():.3f}")
        print(f"                     mean: {Dhs.mean():.3f}")
        print(f"                     max: {Dhs.max():.3f}")

    validate_csv(csv_path=CSV, vtp_path=VTP)

if __name__ == "__main__":
    main()