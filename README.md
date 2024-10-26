# Implementation of nonlinear process monitoring based on random Bernoulli principal component analysis



## Data


The Server Machine Dataset (SMD) from [https://github.com/NetManAIOps/OmniAnomaly](https://github.com/NetManAIOps/OmniAnomaly) was used for this work. The Tennessee Eastman Process (TEP) dataset from [https://github.com/camaramm/tennessee-eastman-profBraatz](https://github.com/camaramm/tennessee-eastman-profBraatz) was used for simulation.


## Code

 - "**RBPCA_modeling.m**" & "**RBPCA_online.m**": The modeling stage and online monitoring stage of static process monitoring based on random Bernoulli PCA.
 - "**TWO_RBPCA_modeling.m**" & "**TWO_RBPCA_online.m**": The modeling stage and online monitoring stage of the dynamic process monitoring based on two-dimensional random Bernoulli PCA.
 - "**MV_RBPCA_modeling.m**" & "**MV_RBPCA_online.m**": The modeling stage and online monitoring stage of the time-varying process monitoring based on moving-window random Bernoulli PCA.
 - "**UCL.m**": Determine the upper control limit by kernel density estimation.
 - "**simu_NE.m**": Generate samples of the numerical examples and perform process monitoring through three methods: random Bernoulli PCA, dynamic random Bernoulli PCA, and two-dimensional random Bernoulli PCA.
 - "**simu_NE_MW.m**": Generate samples of the numerical examples and perform process monitoring through moving-window random Bernoulli PCA.
 - "**real_SMD.m**": Perform process monitoring through three methods: random Bernoulli PCA, dynamic random Bernoulli PCA, and two-dimensional random Bernoulli PCA.
 - "**real_SMD_MV.m**": Perform process monitoring through moving-window random Bernoulli PCA.



## Workflow

 - Simulation: Run "**simu_NE.m**" and "**simu_NE_MW.m**".
 - Real data: Modify the path of the data, then run "**real_SMD.m**" and "**real_SMD_MW.m**".
