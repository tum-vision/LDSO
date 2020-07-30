#!/usr/bin/env bash

set -e

BUILD_TYPE=Release
NUM_PROC=4

BASEDIR="$PWD"

cd "$BASEDIR/thirdparty/DBoW3"
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE ..
make -j$NUM_PROC

cd "$BASEDIR/thirdparty/g2o"
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE ..
make -j$NUM_PROC

cd "$BASEDIR"
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE -DREPO_DIR=\"$BASEDIR\" ..
make -j$NUM_PROC