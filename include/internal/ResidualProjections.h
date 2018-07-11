#pragma once
#ifndef LDSO_RESIDUAL_PROJECTIONS_H_
#define LDSO_RESIDUAL_PROJECTIONS_H_

#include "NumTypes.h"
#include "internal/GlobalCalib.h"

namespace ldso {

    namespace internal {

        EIGEN_STRONG_INLINE float derive_idepth(
                const Vec3f &t, const float &u, const float &v,
                const int &dx, const int &dy, const float &dxInterp,
                const float &dyInterp, const float &drescale) {
            return (dxInterp * drescale * (t[0] - t[2] * u)
                    + dyInterp * drescale * (t[1] - t[2] * v)) * SCALE_IDEPTH;
        }


        // projection equation:
        // K[u,v]^T = 1/d ( K R K^{-1} [u_p, v_p, 1]^T + K t )
        // compute the Ku, Kv in z=1 plane.
        EIGEN_STRONG_INLINE bool projectPoint(
                const float &u_pt, const float &v_pt,
                const float &idepth,
                const Mat33f &KRKi, const Vec3f &Kt,
                float &Ku, float &Kv) {
            Vec3f ptp = KRKi * Vec3f(u_pt, v_pt, 1) + Kt * idepth;
            Ku = ptp[0] / ptp[2];
            Kv = ptp[1] / ptp[2];
            return Ku > 1.1f && Kv > 1.1f && Ku < wM3G && Kv < hM3G;
        }

        // equation:
        // K[u,v]^T = 1/d ( K R K^{-1} [u_p, v_p, 1]^T + K t ) * rescale
        //                < ------------ ptp --------------- >
        /**
         * Project one point from host to target
         * @param u_pt pixel coordinate in host
         * @param v_pt pixel coordinate in host
         * @param idepth idepth in host
         * @param dx bias in the designed pattern
         * @param dy bias in the designed pattern
         * @param HCalib calib matrix
         * @param R R_TW
         * @param t t_TW
         * @param [out] drescale
         * @param [out] u
         * @param [out] v
         * @param [out] Ku  pixel coordinate in target
         * @param [out] Kv  pixel coordinate in target
         * @param [out] KliP
         * @param [out] new_idepth
         * @return
         */
        EIGEN_STRONG_INLINE bool projectPoint(
                const float &u_pt, const float &v_pt,
                const float &idepth,
                const int &dx, const int &dy,
                shared_ptr<CalibHessian> const &HCalib,
                const Mat33f &R, const Vec3f &t,
                float &drescale, float &u, float &v,
                float &Ku, float &Kv, Vec3f &KliP, float &new_idepth) {
            KliP = Vec3f(
                    (u_pt + dx - HCalib->cxl()) * HCalib->fxli(),
                    (v_pt + dy - HCalib->cyl()) * HCalib->fyli(),
                    1);

            Vec3f ptp = R * KliP + t * idepth;
            drescale = 1.0f / ptp[2];
            new_idepth = idepth * drescale;

            if (!(drescale > 0)) {
                return false;
            }

            u = ptp[0] * drescale;
            v = ptp[1] * drescale;
            Ku = u * HCalib->fxl() + HCalib->cxl();
            Kv = v * HCalib->fyl() + HCalib->cyl();

            return Ku > 1.1f && Kv > 1.1f && Ku < wM3G && Kv < hM3G;
        }

    }

}

#endif // LDSO_RESIDUAL_PROJECTIONS_H_
