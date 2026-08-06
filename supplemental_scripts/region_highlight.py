import numpy as np
import pyvista as pv
from scipy.spatial import KDTree

'''
Usage:
1. Set the paths to the surface and centerline files, and to where you want the segmented vessel surface file to be saved
2. Run the script: 'python region_highlight.py'
3. Open Paraview
4. Open the surface mesh file and reduce the opacity
5. Open the segmented vessel surface file, apply a threshold filter to this file
6. In the Properties tab of the filter, ensure the Scalars is set to Region ID and change the thresholds to control which regions are highlighted. Can use a different colour map if desired.
'''

PALETTE = [
    [0, 255, 255], # Cyan
    [255, 0, 0], # Red
    [0, 255, 0], # Green
    [128, 0, 128], #Purple
    [255, 165, 0], #Orange
    [0, 0, 255], # Blue
    [255, 255, 0], #Yellow
    [255, 20, 147], #Deep Pink
]

BACKGROUND_COLOR = [255, 255, 255] #White for the unmarked mesh

# 1. Load your existing VTK PolyData (.vtp) files
surface = pv.read("/home/kabir/masters_files/Gurnish_cases/Good/A/CaseA_cl_remeshed.vtp")
centerline = pv.read("/home/kabir/masters_files/Gurnish_cases/Good/A/CaseA_centerline_with_metrics.vtp")
save_path = "Segmented_Vessels/CaseA_segmented_vessel_surface.vtp"

# 2. Define your regions using centerline point indices
# Create an array of zeros (0 will represent the "unmarked/grey" background)
centerline_regions = np.zeros(centerline.n_points, dtype=int)

# Define your start and end centerline indices for each region
# regions = {
#     1: (20, 45),  # Region ID 1: from index 20 to 45
#     2: (55, 60),  # Region ID 2: from index 46 to 60
#     3: (65, 90),  # Region ID 3: from index 65 to 90
#     4: (110, 140),  # Region ID 4
#     5: (150, 180),  # Region ID 5
# }

#Alternative region creation
regions = {}
for i in range(1, centerline.n_points):
    regions[i] = (i-1, i)

for region_id, (start_idx, end_idx) in regions.items():
    centerline_regions[start_idx : end_idx + 1] = region_id

# Add this array to the centerline object
centerline["RegionID"] = centerline_regions

# 3. Map the centerline regions to the surface mesh using a KD-Tree
# This finds the closest centerline point for every single vertex on the surface mesh
tree = KDTree(centerline.points)
distances, closest_centerline_indices = tree.query(surface.points)
surface_region_ids = centerline["RegionID"][closest_centerline_indices]

# Assign the RegionID from the closest centerline point to the surface mesh points
surface["RegionID"] = surface_region_ids

# 3b. Automatically map regionIDs to the 8-colour palette
rgb_colors = np.zeros((surface.n_points, 3), dtype=np.uint8)
rgb_colors[surface_region_ids == 0] = BACKGROUND_COLOR
for region_id in regions.keys():
    palette_idx = (region_id - 1) % len(PALETTE)
    color = PALETTE[palette_idx]

    rgb_colors[surface_region_ids == region_id] = color

surface.point_data.set_array(rgb_colors, "RGB")
surface.point_data.active_scalars_name = "RGB"

surface.GetPointData().SetActiveScalars("RGB")

# 4. Save the newly tagged surface mesh
surface.save(save_path)
print("Successfully generated segmented_vessel_surface.vtp!")