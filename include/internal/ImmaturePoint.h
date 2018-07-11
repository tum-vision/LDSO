#pragma once
#ifndef LDSO_IMMATURE_POINT_H_
#define LDSO_IMMATURE_POINT_H_

#include "Feature.h"
#include "internal/Residuals.h"
#include "internal/CalibHessian.h"

namespace ldso {


    namespace internal {

        /**
         * the residual of immature point for solving optimization problems on immature points
         * will be converted into a normal map point residual if the immature point turns out to be a good point
         */
        struct ImmaturePointTemporaryResidual {
        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
            ResState state_state;
            double state_energy;
            ResState state_NewState;
            double state_NewEnergy;
            weak_ptr<FrameHessian> target;
        };

        /**
         * immature point status
         */
        enum ImmaturePointStatus {
            IPS_GOOD = 0,               // traced well and good
            IPS_OOB,                    // OOB: end tracking & marginalize!
            IPS_OUTLIER,                // energy too high: if happens again: outlier!
            IPS_SKIPPED,                // traced well and good (but not actually traced).
            IPS_BADCONDITION,           // not traced because of bad condition.
            IPS_UNINITIALIZED           // not even traced once.
        };

        /**
         * The immature point
         * An immature point is a point whose inverse depth has not converged.
         * When a feature is created we will initialize it with an immature point, and then we will try to trace it in other images by searching the epipolar line, and hope it will finally converge. If so, we will create a map point according to this immature point, otherwise we just discard it.
         */
        class ImmaturePoint {
        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

            /**
             * create an immature point from a host frame and a host feature
             * @param hostFrame
             * @param hostFeat
             * @param type
             * @param HCalib
             */
            ImmaturePoint(shared_ptr<Frame> hostFrame, shared_ptr<Feature> hostFeat, float type,
                          shared_ptr<CalibHessian> &HCalib);

            ~ImmaturePoint() {}

            /**
             * Trace the immature point in a new frame
             * @param frame
             * @param hostToFrame_KRKi
             * @param hostToFrame_Kt
             * @param hostToFrame_affine
             * @param HCalib
             * @param debugPrint
             * @return
             */
            ImmaturePointStatus
            traceOn(shared_ptr<FrameHessian> frame, const Mat33f &hostToFrame_KRKi, const Vec3f &hostToFrame_Kt,
                    const Vec2f &hostToFrame_affine, shared_ptr<CalibHessian> HCalib);

            /**
             * compute the energy of the residuals and jacobians
             * @param HCalib
             * @param outlierTHSlack
             * @param tmpRes
             * @param Hdd
             * @param bd
             * @param idepth
             * @return
             */
            double linearizeResidual(
                    shared_ptr<CalibHessian> HCalib, const float outlierTHSlack,
                    shared_ptr<ImmaturePointTemporaryResidual> tmpRes,
                    float &Hdd, float &bd,
                    float idepth);

            float getdPixdd(
                    shared_ptr<CalibHessian> HCalib,
                    shared_ptr<ImmaturePointTemporaryResidual> tmpRes,
                    float idepth);

            float calcResidual(
                    shared_ptr<CalibHessian> HCalib, const float outlierTHSlack,
                    shared_ptr<ImmaturePointTemporaryResidual> tmpRes,
                    float idepth);

            // data
            shared_ptr<Feature> feature = nullptr;    // the feature hosting this immature point

            float color[MAX_RES_PER_POINT];
            float weights[MAX_RES_PER_POINT];

            Mat22f gradH;
            Vec2f gradH_ev;
            Mat22f gradH_eig;
            float energyTH;
            int idxInImmaturePoints;

            float quality = 10000;
            float my_type = 1;

            float idepth_min = 0;
            float idepth_max = NAN;

            ImmaturePointStatus lastTraceStatus = ImmaturePointStatus::IPS_UNINITIALIZED;
            Vec2f lastTraceUV;
            float lastTracePixelInterval;
        };


    }

}

#endif