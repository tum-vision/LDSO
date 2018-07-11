#pragma once
#ifndef LDSO_POINT_HESSIAN_H_
#define LDSO_POINT_HESSIAN_H_

#include "Point.h"
#include "Settings.h"
#include "internal/Residuals.h"
#include "Feature.h"

namespace ldso {
    namespace internal {
        class PointFrameResidual;

        class ImmaturePoint;

        /**
         * Point hessian is the internal structure of a map point
         */
        class PointHessian {
        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

            // create the point hessian from immature point
            PointHessian(shared_ptr<ImmaturePoint> rawPoint);

            PointHessian() {}

            inline void setIdepth(float idepth) {
                this->idepth = idepth;
                this->idepth_scaled = SCALE_IDEPTH * idepth;
                if (point->mHostFeature.expired()) {
                    LOG(FATAL) << "host feature expired!" << endl;
                }
                point->mHostFeature.lock()->invD = idepth;
            }

            inline void setIdepthScaled(float idepth_scaled) {
                this->idepth = SCALE_IDEPTH_INVERSE * idepth_scaled;
                this->idepth_scaled = idepth_scaled;
                if (point->mHostFeature.expired()) {
                    LOG(FATAL) << "host feature expired!" << endl;
                }
                point->mHostFeature.lock()->invD = idepth;
            }

            inline void setIdepthZero(float idepth) {
                idepth_zero = idepth;
                idepth_zero_scaled = SCALE_IDEPTH * idepth;
                nullspaces_scale = -(idepth * 1.001 - idepth / 1.001) * 500;
            }

            // judge if this point is out of boundary
            inline bool isOOB(std::vector<shared_ptr<FrameHessian>> &toMarg) {

                int visInToMarg = 0;
                for (shared_ptr<PointFrameResidual> &r : residuals) {
                    if (r->state_state != ResState::IN) continue;
                    for (shared_ptr<FrameHessian> k : toMarg)
                        if (r->target.lock() == k) visInToMarg++;
                }

                if ((int) residuals.size() >= setting_minGoodActiveResForMarg &&
                    numGoodResiduals > setting_minGoodResForMarg + 10 &&
                    (int) residuals.size() - visInToMarg < setting_minGoodActiveResForMarg)
                    return true;

                if (lastResiduals[0].second == ResState::OOB)
                    return true;
                if (residuals.size() < 2) return false;
                if (lastResiduals[0].second == ResState::OUTLIER && lastResiduals[1].second == ResState::OUTLIER)
                    return true;
                return false;
            }

            inline bool isInlierNew() {
                return (int) residuals.size() >= setting_minGoodActiveResForMarg
                       && numGoodResiduals >= setting_minGoodResForMarg;
            }


            shared_ptr<Point> point = nullptr;

            float u = 0, v = 0;              // pixel position
            float energyTH = 0;
            bool hasDepthPrior = false;
            float my_type = 0;
            float idepth_scaled = 0;
            float idepth_zero_scaled = 0;
            float idepth_zero = 0;
            float idepth = 0;
            float step = 0;
            float step_backup;
            float idepth_backup;
            float nullspaces_scale;
            float idepth_hessian = 0;
            float maxRelBaseline = 0;
            int numGoodResiduals = 0;

            // residuals in many keyframes
            std::vector<shared_ptr<PointFrameResidual>> residuals;   // only contains good residuals (not OOB and not OUTLIER). Arbitrary order.

            // the last two residuals
            std::pair<shared_ptr<PointFrameResidual>, ResState> lastResiduals[2];  // contains information about residuals to the last two (!) frames. ([0] = latest, [1] = the one before).

            // static values
            float color[MAX_RES_PER_POINT];         // colors in host frame
            float weights[MAX_RES_PER_POINT];       // host-weights for respective residuals.

            // ======================================================================== 、、
            // optimization data

            void takeData() {
                priorF = hasDepthPrior ? setting_idepthFixPrior * SCALE_IDEPTH * SCALE_IDEPTH : 0;
                if (setting_solverMode & SOLVER_REMOVE_POSEPRIOR)
                    priorF = 0;
                deltaF = idepth - idepth_zero;
            }

            float priorF = 0;
            float deltaF = 0;

            // H and b blocks
            float bdSumF = 0;
            float HdiF = 0;
            float Hdd_accLF = 0;
            VecCf Hcd_accLF = VecCf::Zero();
            float bd_accLF = 0;
            float Hdd_accAF = 0;
            VecCf Hcd_accAF = VecCf::Zero();
            float bd_accAF = 0;
            bool alreadyRemoved = false;
        };

    }

}

#endif // LDSO_POINT_HESSIAN_H_
