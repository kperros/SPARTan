# SPARTan
The current repository provides the code accompanying the KDD 2017 paper "SPARTan: Scalable PARAFAC2 for Large & Sparse Data",
by Ioakeim Perros, Evangelos E. Papalexakis, Fei Wang, Richard Vuduc, Elizabeth Searles, Michael Thompson and Jimeng Sun.

The entry point is the file quick_start_demo.m.

We have tested the code on MatlabR2015b. The prerequisite packages to run it are:
1) The Tensor Toolbox Version 2.6 (can be downloaded from: http://www.sandia.gov/~tgkolda/TensorToolbox/index-2.6.html)
2) The N-Way Toolbox Version 3.30 (can be downloaded from: http://www.models.life.ku.dk/nwaytoolbox/download)

Also, in order to use the parallel pool capabilities of Matlab, the Parallel Computing Toolbox has to be installed.
Finally, we accredit the dense PARAFAC2 implementation by Rasmus Bro (http://www.models.life.ku.dk/algorithms), from where we have adapted many functionalities.

List of files included:
quick_start_demo.m
create_parafac2_problem.m
parafac2_sparse_paper_version.m
cp_als_for_parafac2.m
cp_als_for_parafac2_baseline.m
mttkrp_for_parafac2.m
mttkrp_mode1.m
mttkrp_mode2.m
mttkrp_mode3.m
unique_col_ind.m
parafac2_fit.m
