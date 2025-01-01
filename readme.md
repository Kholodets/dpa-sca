# Differential Power Analysis Side-channel attack

This repo is simply an archive of the files I used for the analysis of the power traces and cipher texts found in the two directories in this repo to recover the AES key (usually just the 10th round key).
The main analysis happens in the `cpa_experiment.Rmd` file. I used Rstudio for my environment.
Three functions (hypothetical power generation, correlation matrix calculation, and low pass filter) have been written in CUDA and will be loaded by the code in the Rmd file.
I compiled the CUDA files with commands like the following, architecture target may need to be adjusted depending on your system (I have a 3080 TI).
```
$ nvcc -arch=sm_86 -I/usr/include/R/ -L/usr/lib/R/lib -lR --shared -Xcompiler -fPIC -o pcorr.so cucorr.cu
```
