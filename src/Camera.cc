#include "Camera.h"
#include "internal/CalibHessian.h"

using namespace ldso::internal;

namespace ldso {
    Camera::Camera( double fx, double fy, double cx, double cy) {
        this->fx = fx;
        this->fy = fy;
        this->cx = cx;
        this->cy = cy;
    }

    void Camera::CreateCH(shared_ptr<Camera> cam) {
        this->mpCH = shared_ptr<CalibHessian>( new CalibHessian(cam) );
    }

    void Camera::ReleaseCH() {
        if ( mpCH ) {
            mpCH->camera = nullptr;
            mpCH = nullptr;
        }
    }

}