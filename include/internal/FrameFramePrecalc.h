#pragma once
#ifndef LDSO_FRAME_FRAME_PRECALC_H_
#define LDSO_FRAME_FRAME_PRECALC_H_

#include "NumTypes.h"
#include "internal/FrameHessian.h"
#include "internal/CalibHessian.h"

#include <memory>

using namespace std;

namespace ldso {

    namespace internal {

        /**
         * in the inverse depth parameterized bundle adjustment, an observation is related with two frames: the host and the target
         * but we just need to compute once to each two frame pairs, not each observation
         * this structure is used for this precalculation
         */
        struct FrameFramePrecalc {
        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

            void Set(shared_ptr<FrameHessian> host, shared_ptr<FrameHessian> target, shared_ptr<CalibHessian> HCalib);

            weak_ptr<FrameHessian> host; // defines row
            weak_ptr<FrameHessian> target;   // defines column

            // precalc values

            // T_TW * T_WH = T_TH
            // T=target, H=Host
            Mat33f PRE_RTll = Mat33f::Zero();
            Mat33f PRE_RTll_0 = Mat33f::Zero();
            Vec3f PRE_tTll = Vec3f(0, 0, 0);
            Vec3f PRE_tTll_0 = Vec3f(0, 0, 0);
            Mat33f PRE_KRKiTll = Mat33f::Zero();
            Mat33f PRE_RKiTll = Mat33f::Zero();
            Vec2f PRE_aff_mode = Vec2f(0, 0);
            float PRE_b0_mode = 0;
            Vec3f PRE_KtTll = Vec3f(0, 0, 0);
            float distanceLL = 0;
        };
    }
}

#endif // LDSO_FRAME_FRAME_PRECALC_H_
