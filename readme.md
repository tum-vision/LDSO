# minimal instructions for compilation and running
## dependencies
```sudo apt-get install libopencv-dev```

and also install Pangolin for visualization.

## compile
```./make_project.sh```

or follow the steps in this sh file (will compile DBoW3 and g2o first, and the ldso). 

## run  
- TUM-MONO

```./bin/run_dso_dataset``

- EUROC

```./bin/run_dso_euroc```

configure the parameter inside the 'run_dso/run_dso_dataset.cc' or pass arguments like dso: --file=xxx --calib=xxx etc. 

## Notes
- I remove the lib zip dependency so you need to unzip the image files.
- The result_trajectory.zip are trajectories estimated in TUM-Mono. The Matlab_Evaluation.tar.gz is matlab evaluation scripts (check the dataset page of vision.in.tum.de for the original scripts). 
