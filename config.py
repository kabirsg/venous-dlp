#Hydraulic_diameter.py
hd_stl = "" #Path to STL surface mesh file
hd_vtp = "" #Path to VTP file containing centerline points 
hd_csv = "" #Output path for created CSV file with calculated values at each centerline point
hd_viz_dir = "" #Output path to created folder for visualization files - 3 groups of VTP files outputted here (1 for every point): centerline points, planes, outlines

#DLP.py
dlp_blood_dyn_visc = 0.04 #dynamic viscosity mu value [Poise]
dlp_inlet_flow_rate = 5.58 #mL/s - same as Back to Bernoulli paper
dlp_kt = 1.52 #Same as Mirramezani paper
dlp_density = 1.06 #g/mL or g/cm^3
dlp_reynolds_number = 300 #Reynold's number for cerebral venous system - 300 is a placeholder value for now
    
dlp_exp_term = 3 #0 (no expansion), 1(exp res all at one point), 2(exp res applied linearly), 3(exp res applied proportional to radius), 4(exp res applied proportional to area)
dlp_curv_term = 1 #0 - no curvature resistance term added, 1 - curvature resistance term added

dlp_cline_file_path = "" #Path to centerline file
dlp_fig_save_folder = "" #Path to folder where the figures should be saved
dlp_debug = False #Flag to output verbose results to debug text file specified below
dlp_debug_file_path = "" #File path for debug info file - will be created here if it doesn't already exist
dlp_debug_case_name = "" #A name to identify the case in the debug info file

#visualization.py
v_stl = "" #Input STL file
v_cline_vtp_with_metrics = "" #Path to the saved centerline file WITH METRICS - created in hydraulic_diameter.py, same name as centerline vtp file with _with_metrics appended
v_out_surface_vtp = "" # The created output file will be a VTP file so it can store the scalar data arrays