# venous-dlp
Cerebral Venous System Distributed Lumped Parameter Model

Based on the Mirramezani et al. "A Distributed Lumped Parameter Model of Blood Flow" which estimated arterial blood flow in a reduced-order model.
This repository is to validate this DLP for cerebral venous geometries.

### DLP
Required packages for DLP solver package versions that have been tested are denoted after the equal sign. Ex. python=3.14.4:
- python=3.14.4
- vtk=9.5.2
- numpy=2.4.3
- pandas=3.0.2
- scipy=1.17.1
- matplotlib=3.10.9

### Hydraulic Diameter

The hydraulic_diameter.py script calculates the area, perimeter, and hydraulic diameter at each centerline point (given a centerline file from Geometry Tools)
Additional packages required for hydraulic_diameter calculation script:
- pyvista=0.47.3

### Visualization

The visualization.py script maps the the 1D area, perimeter, and hydraulic diameter metrics back on to a 3D surface (saved as a vtp file) - viewable in Paraview

### Bernoulli
Replicating the Bernoulli data from Gurnish Sidora's Back to Bernoulli (2025) paper

## Usages

Usage of DLP solver:
1. Set config.py file with the necessary file paths, constants, and the terms that you want included in the solver
2. Run `python DLP.py`

Usage of hydraulic diameter, area, and perimeter calculation script:
1. Set file paths and constant values (caps variables) at top of main function
2. Run `python hydralic_diameter.py`

Usage of visualization.py:
1. Set file paths in config.py file
2. Run `python visualization.py`

The differences between DLP scripts within this repository, with each building on the last:  

| File | Description |
|:----|:----|
| **DLP.py:** | <ul><li>First attempt at the simple DLP (not used anymore)| 
| **DLP_Dh.py:** | <ul><li>Uses hydraulic diameter for calculations and adds the curvature term - requires the hydraulic diameter script to be run beforehand | 
| **DLP_v3.py:** | <ul><li>Uses a different method for finding the local mins and maxes <li>only has linear distribution for the expansion resistances within the expansion region <li>discretizes the vessel in a backwards method so the length segment for point i is the distance between points i and i-1 <li>changes curvature calculation from using instantaneous curvature on a point-by-point basis to a rolling average approach for curvature|
| **DLP_v4.py:** | <ul><li> Hybrid DLP + Bernoulli solver based on regions where each term gives a more accurate solution|