#include "Settings.h"
#include "internal/PointHessian.h"
#include "internal/ImmaturePoint.h"

namespace ldso {

    namespace internal {

        PointHessian::PointHessian(shared_ptr<ImmaturePoint> rawPoint) {
            u = rawPoint->feature->uv[0];
            v = rawPoint->feature->uv[1];
            my_type = rawPoint->my_type;

            this->idepth = SCALE_IDEPTH_INVERSE * (rawPoint->idepth_max + rawPoint->idepth_min) * 0.5;
            this->idepth_scaled = this->idepth;

            int n = patternNum;
            memcpy(color, rawPoint->color, sizeof(float) * n);
            memcpy(weights, rawPoint->weights, sizeof(float) * n);
            energyTH = rawPoint->energyTH;
        }
    }

}
