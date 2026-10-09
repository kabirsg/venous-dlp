############################
### DLP.py and DLP_Dh.py ###
############################

dlp_blood_dyn_visc = 0.037 #dynamic viscosity mu value [Poise]
dlp_inlet_flow_rate = 3.73 #[mL/s] - Patient-specific flow rate from Table 2 of the Back to Bernoulli Paper
dlp_kt = 1.52 #Same as Mirramezani paper
dlp_density = 1.06 #[g/mL] or [g/cm^3]

dlp_exp_term = 1 #1(exp res applied proportionally to radius), 2(exp res proportional to area), 3(exp res inversely proportional to radius)
dlp_inlet_point_idx = 0 #0 means that the inlet that was designated in the centerline file is the actual inlet. -1 for if the inlet and outlet are inversed

# dlp_cline_file_path = "/home/kabir/masters_files/Gurnish_cases/Good/I/CaseI_centerline_with_metrics.vtp" #Path to centerline file
# dlp_fig_save_folder = "outputs" #Path to folder where the figures should be saved
# dlp_case_name = "CaseF" #Name to identify the case in the debug info file
# dlp_cfd_case_name = "Case I individual" #Case name in the CFD data tecplot file

dlp_cline_file_path ="/home/kabir/masters_files/Gurnish_cases/Good/A/CaseA_centerline_with_metrics.vtp" #Path to centerline file
dlp_fig_save_folder = "outputs" #Path to folder where the figures should be saved
dlp_case_name = "CaseA"
dlp_cfd_case_name = "Case A individual"

# dlp_cline_file_path ="/home/kabir/masters_files/Cases_for_Rojin/eccentric_stenosCaseA_default_centreline_with_metricsis_d_10/ecc_stenosis_v10/eccStenosis_cl_centerline_graph_vmtk_with_metrics.vtp" #Path to centerline file
# dlp_surf_file_path = "/home/kabir/masters_files/Cases_for_Rojin/eccentric_stenosis_d_10/ecc_stenosis_v10/eccStenosis_cl_remeshed.vtp"
# dlp_fig_save_folder = "/home/kabir/masters_files/Cases_for_Rojin/eccentric_stenosis_d_10/ecc_stenosis_v10/eccStenosis_v10_dlp_fig_save" #Path to folder where the figures should be saved
# dlp_case_name = "eccentric_stenosis"