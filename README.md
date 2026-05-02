# venous-dlp
Cerebral Venous System Distributed Lumped Parameter Model

Based on the Mirramezani et al. "A Distributed Lumped Parameter Model of Blood Flow" which estimated arterial blood flow in a reduced-order model.
This repository is to validate this DLP for cerebral venous geometries.

Required packages for DLP solver package versions that have been tested are denoted after the equal sign. Ex. python=3.14.4:
- vtk=9.5.2
- numpy=2.4.3
- pandas=3.0.2
- scipy=1.17.1
- matplotlib

Additional packages required for hydraulic_diameter calculation script:
- rtree=1.4.1
- trimesh=4.12.1
- pyvista=0.47.3

Usage of DLP solver:
1. Set config.py file with the necessary file paths
2. Set constants and which terms you want in the solver in the main function 
2. Run `python DLP.py`

Usage of hydraulic diameter, area, and perimeter calculation script:
1. Set file paths and constant values (caps variables) at top of main function
2. Run `python hydralic_diameter.py`