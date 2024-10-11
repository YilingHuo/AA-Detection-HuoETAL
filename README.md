# Code-for-Huo-et-al.-2024-
Code for creating figures in Huo et al. (2024)
----
## Citation
If you use this code or adapt it in your work, please cite the following paper:

Huo, Y., Wang, H., Lu, J., Fu, Q., Jonko, A.K., Lee, Y.J., Ma, W., Maslowski, W. & Qin, Y., 2024. Assessing Radiative Feedbacks and Their Contribution to the Arctic Amplification Measured by Various Metrics. *Journal of Geophysical Research: Atmosphere*, 129, e2024JD040880. https://doi.org/10.1029/e2024JD040880
---- 
## Quick Start
To create Figs.1-3 in Huo et al. (2024) run Fig1.py, Fig2.py and Fig3.py respectively.
For Figs. 4-7, you'll need NCL, NCO and Python.
1. Download the CAM5 kernels (alb.kernel.nc, q.kernel.nc, t.kernel.nc, ts.kernel.nc) and radiative forcings (aerosol.forcing.nc, ghg.forcing.nc) here: https://zenodo.org/record/997902 
2. Download the NCL codes (q_kernel_to_plev.ncl, t_kernel_to_plev.ncl) to convert the kernels from hybrid sigma-pressure to standard pressure vertical coordinate from here https://github.com/apendergrass/cam5-kernels/blob/master/tools  
`ncl tools/t_kernel_to_plev.ncl`  
`ncl tools/q_kernel_to_plev.ncl`
3. use ncremap to regrid data files to the same as kernel files. 
4. run planck_lapserate_albedo_watervapor.py and cloud_feedback.py first to create the text files of zonal mean radiative feedbacks.
5. Run Fig4.py, Fig5.py, Fig6.py and Fig7.py respectively to create Figs.4-7 in Huo et al. (2024).
-----
Questions/comments? Post in the issues tab, or email me at yiling.huo at pnnl dot com
