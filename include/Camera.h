#pragma once
#ifndef LDSO_CAMERA_H_
#define LDSO_CAMERA_H_

#include "NumTypes.h"

namespace ldso {

    namespace internal {
        class CalibHessian;
    }

    /**
     * @brief Pinhole camera model
     * the parameters will be estimated during optimization
     */
    struct Camera {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        Camera(double fx, double fy, double cx, double cy);

        /**
         * Create the internal structure, otherwise it will be nullptr
         * @param cam
         */
        void CreateCH(shared_ptr<Camera> cam);

        /**
         * Release the internal structure
         */
        void ReleaseCH();

        // data
        double fx = 0, fy = 0, cx = 0, cy = 0;

        // internal structure
        shared_ptr<internal::CalibHessian> mpCH = nullptr;
    };

}

#endif // LDSO_CAMERA_H_
