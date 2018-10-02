# LDSO: Direct Sparse Odometry with Loop Closure

## Related Publications

 * **LDSO: Direct Sparse Odometry with Loop Closure**, *X. Gao, R. Wang, N. Demmel, D. Cremers*,
   In International Conference on Intelligent Robots and Systems (IROS), 2018.
 * **Direct Sparse Odometry**, *J. Engel, V. Koltun, D. Cremers*,
   In IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI), 2018
 * **A Photometrically Calibrated Benchmark For Monocular Visual Odometry**, *J. Engel, V. Usenko, D. Cremers*,
   In arXiv:1607.02555, 2016

## Dependencies

### System dependencies

There is a convenience script that will help you install the needed
libraries in Ubuntu 16.04 and later, including `Eigen`, `glog`,
`gtest`, `Suitesparse`, `OpenCV`, `libzip`.

```
./install_dependencies.sh
```

On OSX you can install these via Homebrew.

### Other libraries

Compile and install
[Pangolin](https://github.com/stevenlovegrove/Pangolin) for
visualization.
 
### Compile
 
 ```
 ./make_project.sh
 ```

This will build the thirdparty library and also ldso library for
you. You can also follow the steps in this script manually (will
compile DBoW3 and g2o first, and the ldso).

## Run

We provide examples on three datasets.
 - TUM-Mono: [https://vision.in.tum.de/mono-dataset](https://vision.in.tum.de/mono-dataset)
 - Kitti: [Kitti odometry](http://www.cvlibs.net/datasets/kitti/eval_odometry.php)
 - EuRoC: [Euroc MAV dataset](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets)

You can easily extend them in the `examples` folder or add your own
executables for your camera or dataset.

After compilation, in the `bin` directory there will be three
executables like `run_dso_xxx`. You can either specify the directories
in the source file like `examples/run_dso_xxx.cc`, or pass them as
command line parameters. When running LDSO, you will see a window
showing an visualization of camera trajectory and tracked points.

Make sure your working directory is at the root of LDSO code, to
ensure that files like the BoW vocabulary file are found.

**TUM-Mono:**

To run LDSO on TUM-Mono dataset sequence 34, execute:

```
./bin/run_dso_tum_mono \
    preset=0 \
    files=XXXXX/TUMmono/sequences/sequence_34/images.zip \
    vignette=XXXXX/TUMmono/sequences/sequence_34/vignette.png \
    calib=XXXXX/TUMmono/sequences/sequence_34/camera.txt \
    gamma=XXXXX/TUMmono/sequences/sequence_34/pcalib.txt
```

**Kitti:**

To run LDSO on Kitti dataset sequence 00, execute:

```
./bin/run_dso_kitti \
    preset=0 \
    files=XXXXX/Kitti/odometry/dataset/sequences/00/ \
    calib=./examples/Kitti/Kitti00-02.txt
```

**EuRoC:**

To run LDSO on EuRoC dataset sequence MH_01_easy, execute:

```
./bin/run_dso_euroc \
    preset=0 \
    files=XXXX/EuRoC/MH_01_easy/mav0/cam0/
```

## Notes

 - LDSO is a monocular VO based on DSO with Sim(3) loop closing
   function. Note we still **cannot** know the real scale of
   mono-slam. We only make it more consistent in long trajectories.
   
 - The red line in pangolin windows shows the trajectory before loop
   closing, and the yellow line shows the trajectory after
   optimization.
   
 - If you are looking for code instructions, take a look at
   `doc/notes_on_ldso.pdf` and see if it can help you.
 
 - Set `setting_enableLoopClosing` to true/false to turn on/off loop
   closing function. `setting_fastLoopClosing` will record less data and
   make the loop closing faster.
   
 - If you need loop closing, please set `setting_pointSelection=1` to
   make the program compute feature descriptors. If
   `setting_pointSelection=0`, the program acts just like DSO, and
   `setting_pointSelection=2` means random point selection, which is
   faster but unstable.
   
 - Some of the GUI buttons may not work.
   
 - You might also want to have a look DSO's README:
   https://github.com/JakobEngel/dso/blob/master/README.md

## License

LDSO, like DSO, is licensed under GPLv3. It makes use of several
third-party libraries. Among other's it comes with the source of:
 - Sophus ([MIT](https://github.com/strasdat/Sophus/blob/master/LICENSE.txt))
 - g2o ([BSD, GPL, LGPL, ...](https://github.com/RainerKuemmerle/g2o/blob/master/README.md))
 - DBoW3 ([BSD](https://github.com/raulmur/ORB_SLAM2/blob/master/Thirdparty/DBoW2/LICENSE.txt))
