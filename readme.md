# LDSO: Direct Sparse Odometry with Loop Closure
### 1. Related Ppaers
* **LDSO: Direct Sparse Odometry with Loop Closure**, X. Gao, R. Wang, N. Demmel, D. Cremers, In International Conference on Intelligent Robots and Systems (IROS), 2018.
* **Direct Sparse Odometry**, *J. Engel, V. Koltun, D. Cremers*, In arXiv:1607.02565, 2016
* **A Photometrically Calibrated Benchmark For Monocular Visual Odometry**, *J. Engel, V. Usenko, D. Cremers*, In arXiv:1607.02555, 2016

# minimal instructions for compilation and running
## dependencies
### libraries in Ubuntu
```./install_dependencies.sh```
This will help you install the needed libraries in Ubuntu 16.04 and later, including eigen, glog, opencv, libzip, etc.
### other libraries
 - install [Pangolin](https://github.com/stevenlovegrove/Pangolin) for visualization. 

## compile
```./make_project.sh```

This will build the thirdparty library and also ldso library for you. You can also follow the steps in this sh file (will compile DBoW3 and g2o first, and the ldso). 

## run  
We provide examples on three datasets. TUM-Mono: [https://vision.in.tum.de/mono-dataset](https://vision.in.tum.de/mono-dataset), Kitti: [Kitti odometry](http://www.cvlibs.net/datasets/kitti/eval_odometry.php), and EUROC: [Euroc MAV dataset](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets). You can also write your own executable programs on your camera. 

After compilation, in **bin** directory there will be three executables like run_dso_xxx. You can either specify the directories in the source file like examples/run_dso_xxx.cc, or pass them as command line parameters. For example, if you want to run ldso on Kitti dataset and use sequence 00, please change the paths in examples/run_dso_xxx.cc like:

> std::string source = "/media/Data/Dataset/Kitti/dataset/sequences/00";
std::string output_file = "./results.txt";
std::string calib = "./examples/Kitti/Kitti00-02.txt";
std::string vocPath = "./vocab/orbvoc.dbow3";

So here we assume your images files are store at /media/Data/Dataset/Kitti/dataset/sequences/00 and use the calibration file in ./examples/Kitti/Kitti00-02.txt. Make sure your working directory is at the root of LDSO code, and run:
```./bin/run_dso_kitti```
and then you will see a window showing the status of the program. Do the same for EUROC and TUM-Mono dataset if you want. 

Or, if you don't want to change the code, just call the executables like DSO: 

```bin/run_dso_tum_mono \
			files=XXXXX/sequence_XX/images.zip \
			calib=XXXXX/sequence_XX/camera.txt \
			gamma=XXXXX/sequence_XX/pcalib.txt \
			vignette=XXXXX/sequence_XX/vignette.png \
			preset=0 \
			mode=0 ```

## Notes
 - LDSO is a monocular VO based on DSO with Sim(3) loop closing function. Note we still can not know the real scale of mono-slam. We only make it more consistent in long trajectories. 
 - The red line in pangolin windows shows the trajectory before loop closing, and the yellow line shows the trajectory after optimization.
 - If you are looking for code instructions, take a look at doc/notes_on_ldso.pdf and see if it can help you.
 - The license of LDSO follows DSO. 
