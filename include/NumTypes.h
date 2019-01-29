#pragma once
#ifndef LDSO_NUM_TYPES_H_
#define LDSO_NUM_TYPES_H_

// many numerical types needed in dso
#include <Eigen/Core>
#include <Eigen/StdVector>
#include "sophus/se3.hpp"
#include "sophus/sim3.hpp"

#include <memory>

using namespace std;

#include <glog/logging.h>

#include "DBoW3/src/DBoW3.h"

// pose represented as sophus structs
typedef Sophus::SE3d SE3;
typedef Sophus::SO3d SO3;
typedef Sophus::Sim3d Sim3;

// Camera intrinsics: fx, fy, cx, cy, so dimension =4
// note that the types below which have "C" means camera parameters
const int CPARS = 4;

const int MAX_RES_PER_POINT = 8;  // number of residuals in each point, see dso's paper for the pattern

// double matricies
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatXX;
typedef Eigen::Matrix<double, CPARS, CPARS> MatCC;
typedef Eigen::Matrix<double, CPARS, 10> MatC10;
typedef Eigen::Matrix<double, 10, 10> Mat1010;
typedef Eigen::Matrix<double, 13, 13> Mat1313;
typedef Eigen::Matrix<double, 8, 10> Mat810;
typedef Eigen::Matrix<double, 8, 3> Mat83;
typedef Eigen::Matrix<double, 6, 6> Mat66;
typedef Eigen::Matrix<double, 5, 3> Mat53;
typedef Eigen::Matrix<double, 4, 3> Mat43;
typedef Eigen::Matrix<double, 4, 2> Mat42;
typedef Eigen::Matrix<double, 3, 3> Mat33;
typedef Eigen::Matrix<double, 2, 2> Mat22;
typedef Eigen::Matrix<double, 8, CPARS> Mat8C;
typedef Eigen::Matrix<double, CPARS, 8> MatC8;
typedef Eigen::Matrix<double, 8, 8> Mat88;
typedef Eigen::Matrix<double, 7, 7> Mat77;
typedef Eigen::Matrix<double, 4, 9> Mat49;
typedef Eigen::Matrix<double, 8, 9> Mat89;
typedef Eigen::Matrix<double, 9, 4> Mat94;
typedef Eigen::Matrix<double, 9, 8> Mat98;
typedef Eigen::Matrix<double, 8, 1> Mat81;
typedef Eigen::Matrix<double, 1, 8> Mat18;
typedef Eigen::Matrix<double, 9, 1> Mat91;
typedef Eigen::Matrix<double, 1, 9> Mat19;
typedef Eigen::Matrix<double, 8, 4> Mat84;
typedef Eigen::Matrix<double, 4, 8> Mat48;
typedef Eigen::Matrix<double, 4, 4> Mat44;
typedef Eigen::Matrix<double, 14, 14> Mat1414;
typedef Eigen::Matrix<double, 8 + CPARS + 1, 8 + CPARS + 1> MatPCPC;

// float matricies
typedef Eigen::Matrix<float, 3, 3> Mat33f;
typedef Eigen::Matrix<float, 10, 3> Mat103f;
typedef Eigen::Matrix<float, 2, 2> Mat22f;
typedef Eigen::Matrix<float, 3, 1> Vec3f;
typedef Eigen::Matrix<float, 2, 1> Vec2f;
typedef Eigen::Matrix<float, 6, 1> Vec6f;
typedef Eigen::Matrix<float, 8, CPARS> Mat8Cf;
typedef Eigen::Matrix<float, CPARS, 8> MatC8f;
typedef Eigen::Matrix<float, 1, 8> Mat18f;
typedef Eigen::Matrix<float, 6, 6> Mat66f;
typedef Eigen::Matrix<float, 8, 8> Mat88f;
typedef Eigen::Matrix<float, 8, 4> Mat84f;
typedef Eigen::Matrix<float, 6, 6> Mat66f;
typedef Eigen::Matrix<float, 4, 4> Mat44f;
typedef Eigen::Matrix<float, 12, 12> Mat1212f;
typedef Eigen::Matrix<float, 13, 13> Mat1313f;
typedef Eigen::Matrix<float, 10, 10> Mat1010f;
typedef Eigen::Matrix<float, 9, 9> Mat99f;
typedef Eigen::Matrix<float, 4, 2> Mat42f;
typedef Eigen::Matrix<float, 6, 2> Mat62f;
typedef Eigen::Matrix<float, 1, 2> Mat12f;
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> MatXXf;
typedef Eigen::Matrix<float, 8 + CPARS + 1, 8 + CPARS + 1> MatPCPCf;
typedef Eigen::Matrix<float, 14, 14> Mat1414f;

// double vectors
typedef Eigen::Matrix<double, CPARS, 1> VecC;
typedef Eigen::Matrix<double, 14, 1> Vec14;
typedef Eigen::Matrix<double, 13, 1> Vec13;
typedef Eigen::Matrix<double, 10, 1> Vec10;
typedef Eigen::Matrix<double, 9, 1> Vec9;
typedef Eigen::Matrix<double, 8, 1> Vec8;
typedef Eigen::Matrix<double, 7, 1> Vec7;
typedef Eigen::Matrix<double, 6, 1> Vec6;
typedef Eigen::Matrix<double, 5, 1> Vec5;
typedef Eigen::Matrix<double, 4, 1> Vec4;
typedef Eigen::Matrix<double, 3, 1> Vec3;
typedef Eigen::Matrix<double, 2, 1> Vec2;
typedef Eigen::Matrix<double, Eigen::Dynamic, 1> VecX;
typedef Eigen::Matrix<double, 8 + CPARS + 1, 1> VecPC;

// float vectors
typedef Eigen::Matrix<float, CPARS, 1> VecCf;
typedef Eigen::Matrix<float, MAX_RES_PER_POINT, 1> VecNRf;
typedef Eigen::Matrix<float, 12, 1> Vec12f;
typedef Eigen::Matrix<float, 8, 1> Vec8f;
typedef Eigen::Matrix<float, 10, 1> Vec10f;
typedef Eigen::Matrix<float, 4, 1> Vec4f;
typedef Eigen::Matrix<float, 12, 1> Vec12f;
typedef Eigen::Matrix<float, 13, 1> Vec13f;
typedef Eigen::Matrix<float, 9, 1> Vec9f;
typedef Eigen::Matrix<float, Eigen::Dynamic, 1> VecXf;
typedef Eigen::Matrix<float, 8 + CPARS + 1, 1> VecPCf;
typedef Eigen::Matrix<float, 14, 1> Vec14f;

// unsigned char vectors
typedef Eigen::Matrix<unsigned char, 3, 1> Vec3b;

// Vector of Eigen vectors
typedef vector<Vec2, Eigen::aligned_allocator<Vec2>> VecVec2;
typedef vector<Vec3, Eigen::aligned_allocator<Vec3>> VecVec3;
typedef vector<Vec2f, Eigen::aligned_allocator<Vec2f>> VecVec2f;
typedef vector<Vec3f, Eigen::aligned_allocator<Vec3f>> VecVec3f;

// DBoW vocabulary
typedef DBoW3::Vocabulary ORBVocabulary;


#endif // LDSO_NUM_TYPES_H_
