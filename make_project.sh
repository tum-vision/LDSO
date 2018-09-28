#!/usr/bin/env bash

BUILD_TYPE=Release
NUM_PROC=4

BASEDIR="$PWD"

cd "$BASEDIR/thirdparty/DBoW3"
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE ..
make -j$NUM_PROC

cd "$BASEDIR/thirdparty/g2o"
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE ..
make -j$NUM_PROC

cd "$BASEDIR"
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE ..
make -j$NUM_PROC

