#include "internal/Residuals.h"
#include "internal/FrameHessian.h"
#include "internal/PointHessian.h"
#include "internal/ResidualProjections.h"
#include "internal/GlobalFuncs.h"
#include "Settings.h"
#include "internal/OptimizationBackend/EnergyFunctional.h"

namespace ldso {

    namespace internal {

        double PointFrameResidual::linearize(shared_ptr<CalibHessian> &HCalib) {

            // compute jacobians
            state_NewEnergyWithOutlier = -1;
            if (state_state == ResState::OOB) // 当前状态已经位于外边
            {
                state_NewState = ResState::OOB;
                return state_energy;
            }

            shared_ptr<FrameHessian> f = host.lock();
            shared_ptr<FrameHessian> ftarget = target.lock();
            shared_ptr<PointHessian> fPoint = point.lock();
            FrameFramePrecalc *precalc = &(f->targetPrecalc[ftarget->idx]);

            float energyLeft = 0;
            const Eigen::Vector3f *dIl = ftarget->dI;
            const Mat33f &PRE_KRKiTll = precalc->PRE_KRKiTll;
            const Vec3f &PRE_KtTll = precalc->PRE_KtTll;
            const Mat33f &PRE_RTll_0 = precalc->PRE_RTll_0;
            const Vec3f &PRE_tTll_0 = precalc->PRE_tTll_0;
            const float *const color = fPoint->color;
            const float *const weights = fPoint->weights;

            Vec2f affLL = precalc->PRE_aff_mode;
            float b0 = precalc->PRE_b0_mode;

            // 李代数到xy的导数
            Vec6f d_xi_x, d_xi_y;

            // Calib到xy的导数
            Vec4f d_C_x, d_C_y;

            // xy 到 idepth 的导数
            float d_d_x, d_d_y;

            {
                float drescale, u, v, new_idepth;  // data in target
                // NOTE u = X/Z, v=Y/Z in target
                float Ku, Kv;
                Vec3f KliP;

                // 重投影
                shared_ptr<PointHessian> p = point.lock();
                if (!projectPoint(p->u, p->v, p->idepth_zero_scaled, 0, 0, HCalib,
                                  PRE_RTll_0, PRE_tTll_0, drescale, u, v, Ku, Kv, KliP, new_idepth)) {
                    state_NewState = ResState::OOB;
                    return state_energy;
                }

                centerProjectedTo = Vec3f(Ku, Kv, new_idepth);

                // 各种导数
                // diff d_idepth
                d_d_x = drescale * (PRE_tTll_0[0] - PRE_tTll_0[2] * u) * SCALE_IDEPTH * HCalib->fxl();
                d_d_y = drescale * (PRE_tTll_0[1] - PRE_tTll_0[2] * v) * SCALE_IDEPTH * HCalib->fyl();

                // diff calib
                d_C_x[2] = drescale * (PRE_RTll_0(2, 0) * u - PRE_RTll_0(0, 0));
                d_C_x[3] = HCalib->fxl() * drescale * (PRE_RTll_0(2, 1) * u - PRE_RTll_0(0, 1)) * HCalib->fyli();
                d_C_x[0] = KliP[0] * d_C_x[2];
                d_C_x[1] = KliP[1] * d_C_x[3];

                d_C_y[2] = HCalib->fyl() * drescale * (PRE_RTll_0(2, 0) * v - PRE_RTll_0(1, 0)) * HCalib->fxli();
                d_C_y[3] = drescale * (PRE_RTll_0(2, 1) * v - PRE_RTll_0(1, 1));
                d_C_y[0] = KliP[0] * d_C_y[2];
                d_C_y[1] = KliP[1] * d_C_y[3];

                d_C_x[0] = (d_C_x[0] + u) * SCALE_F;
                d_C_x[1] *= SCALE_F;
                d_C_x[2] = (d_C_x[2] + 1) * SCALE_C;
                d_C_x[3] *= SCALE_C;

                d_C_y[0] *= SCALE_F;
                d_C_y[1] = (d_C_y[1] + v) * SCALE_F;
                d_C_y[2] *= SCALE_C;
                d_C_y[3] = (d_C_y[3] + 1) * SCALE_C;

                // xy到李代数的导数，形式见十四讲
                d_xi_x[0] = new_idepth * HCalib->fxl();
                d_xi_x[1] = 0;
                d_xi_x[2] = -new_idepth * u * HCalib->fxl();
                d_xi_x[3] = -u * v * HCalib->fxl();
                d_xi_x[4] = (1 + u * u) * HCalib->fxl();
                d_xi_x[5] = -v * HCalib->fxl();

                d_xi_y[0] = 0;
                d_xi_y[1] = new_idepth * HCalib->fyl();
                d_xi_y[2] = -new_idepth * v * HCalib->fyl();
                d_xi_y[3] = -(1 + v * v) * HCalib->fyl();
                d_xi_y[4] = u * v * HCalib->fyl();
                d_xi_y[5] = u * HCalib->fyl();
            }


            {
                J->Jpdxi[0] = d_xi_x;
                J->Jpdxi[1] = d_xi_y;

                J->Jpdc[0] = d_C_x;
                J->Jpdc[1] = d_C_y;

                J->Jpdd[0] = d_d_x;
                J->Jpdd[1] = d_d_y;

            }

            float JIdxJIdx_00 = 0, JIdxJIdx_11 = 0, JIdxJIdx_10 = 0;
            float JabJIdx_00 = 0, JabJIdx_01 = 0, JabJIdx_10 = 0, JabJIdx_11 = 0;
            float JabJab_00 = 0, JabJab_01 = 0, JabJab_11 = 0;

            float wJI2_sum = 0;

            for (int idx = 0; idx < patternNum; idx++) {
                float Ku, Kv;
                shared_ptr<PointHessian> p = point.lock();
                if (!projectPoint(p->u + patternP[idx][0], p->v + patternP[idx][1], p->idepth_scaled,
                                  PRE_KRKiTll, PRE_KtTll, Ku, Kv)) {
                    state_NewState = ResState::OOB;
                    return state_energy;
                }

                projectedTo[idx][0] = Ku;
                projectedTo[idx][1] = Kv;

                Vec3f hitColor = (getInterpolatedElement33(dIl, Ku, Kv, wG[0]));
                float residual = hitColor[0] - (float) (affLL[0] * color[idx] + affLL[1]);

                float drdA = (color[idx] - b0);
                if (!std::isfinite((float) hitColor[0])) {
                    state_NewState = ResState::OOB;
                    return state_energy;
                }


                float w = sqrtf(setting_outlierTHSumComponent /
                                (setting_outlierTHSumComponent + hitColor.tail<2>().squaredNorm()));
                w = 0.5f * (w + weights[idx]);

                float hw = fabsf(residual) < setting_huberTH ? 1 : setting_huberTH / fabsf(residual);
                energyLeft += w * w * hw * residual * residual * (2 - hw);

                {
                    if (hw < 1) hw = sqrtf(hw);
                    hw = hw * w;

                    hitColor[1] *= hw;
                    hitColor[2] *= hw;

                    J->resF[idx] = residual * hw;

                    J->JIdx[0][idx] = hitColor[1];
                    J->JIdx[1][idx] = hitColor[2];
                    J->JabF[0][idx] = drdA * hw;
                    J->JabF[1][idx] = hw;

                    JIdxJIdx_00 += hitColor[1] * hitColor[1];
                    JIdxJIdx_11 += hitColor[2] * hitColor[2];
                    JIdxJIdx_10 += hitColor[1] * hitColor[2];

                    JabJIdx_00 += drdA * hw * hitColor[1];
                    JabJIdx_01 += drdA * hw * hitColor[2];
                    JabJIdx_10 += hw * hitColor[1];
                    JabJIdx_11 += hw * hitColor[2];

                    JabJab_00 += drdA * drdA * hw * hw;
                    JabJab_01 += drdA * hw * hw;
                    JabJab_11 += hw * hw;

                    wJI2_sum += hw * hw * (hitColor[1] * hitColor[1] + hitColor[2] * hitColor[2]);

                    if (setting_affineOptModeA < 0) J->JabF[0][idx] = 0;
                    if (setting_affineOptModeB < 0) J->JabF[1][idx] = 0;

                }
            }

            J->JIdx2(0, 0) = JIdxJIdx_00;
            J->JIdx2(0, 1) = JIdxJIdx_10;
            J->JIdx2(1, 0) = JIdxJIdx_10;
            J->JIdx2(1, 1) = JIdxJIdx_11;
            J->JabJIdx(0, 0) = JabJIdx_00;
            J->JabJIdx(0, 1) = JabJIdx_01;
            J->JabJIdx(1, 0) = JabJIdx_10;
            J->JabJIdx(1, 1) = JabJIdx_11;
            J->Jab2(0, 0) = JabJab_00;
            J->Jab2(0, 1) = JabJab_01;
            J->Jab2(1, 0) = JabJab_01;
            J->Jab2(1, 1) = JabJab_11;

            state_NewEnergyWithOutlier = energyLeft;

            if (energyLeft > std::max<float>(f->frameEnergyTH, ftarget->frameEnergyTH) || wJI2_sum < 2) {
                energyLeft = std::max<float>(f->frameEnergyTH, ftarget->frameEnergyTH);
                state_NewState = ResState::OUTLIER;
            } else {
                state_NewState = ResState::IN;
            }

            state_NewEnergy = energyLeft;
            return energyLeft;
        }

        void PointFrameResidual::fixLinearizationF(shared_ptr<EnergyFunctional> ef) {

            Vec8f dp = ef->adHTdeltaF[hostIDX + ef->nFrames * targetIDX];

            // compute Jp*delta
            __m128 Jp_delta_x = _mm_set1_ps(J->Jpdxi[0].dot(dp.head<6>())
                                            + J->Jpdc[0].dot(ef->cDeltaF)
                                            + J->Jpdd[0] * point.lock()->deltaF);
            __m128 Jp_delta_y = _mm_set1_ps(J->Jpdxi[1].dot(dp.head<6>())
                                            + J->Jpdc[1].dot(ef->cDeltaF)
                                            + J->Jpdd[1] * point.lock()->deltaF);

            __m128 delta_a = _mm_set1_ps((float) (dp[6]));
            __m128 delta_b = _mm_set1_ps((float) (dp[7]));

            for (int i = 0; i < patternNum; i += 4) {
                // PATTERN: rtz = resF - [JI*Jp Ja]*delta.
                __m128 rtz = _mm_load_ps(((float *) &J->resF) + i);
                rtz = _mm_sub_ps(rtz, _mm_mul_ps(_mm_load_ps(((float *) (J->JIdx)) + i), Jp_delta_x));
                rtz = _mm_sub_ps(rtz, _mm_mul_ps(_mm_load_ps(((float *) (J->JIdx + 1)) + i), Jp_delta_y));
                rtz = _mm_sub_ps(rtz, _mm_mul_ps(_mm_load_ps(((float *) (J->JabF)) + i), delta_a));
                rtz = _mm_sub_ps(rtz, _mm_mul_ps(_mm_load_ps(((float *) (J->JabF + 1)) + i), delta_b));
                _mm_store_ps(((float *) &res_toZeroF) + i, rtz);
            }

            isLinearized = true;
        }

        /*
        double FeatureObsResidual::linearize(shared_ptr<CalibHessian> &HCalib) {

            // compute jacobians
            state_NewEnergyWithOutlier = -1;
            if (state_state == ResState::OOB) { // 当前状态已经位于外边
                state_NewState = ResState::OOB;
                return state_energy;
            }

            shared_ptr<FrameHessian> f = host.lock();
            shared_ptr<FrameHessian> ftarget = target.lock();
            shared_ptr<PointHessian> fPoint = point.lock();

            FrameFramePrecalc *precalc = &(f->targetPrecalc[ftarget->idx]);

            float energyLeft = 0;
            const Eigen::Vector3f *dIl = ftarget->dI;
            const Mat33f &PRE_KRKiTll = precalc->PRE_KRKiTll;
            const Vec3f &PRE_KtTll = precalc->PRE_KtTll;
            const Mat33f &PRE_RTll_0 = precalc->PRE_RTll_0;
            const Vec3f &PRE_tTll_0 = precalc->PRE_tTll_0;
            const float *const color = fPoint->color;
            const float *const weights = fPoint->weights;

            Vec2f affLL = precalc->PRE_aff_mode;
            float b0 = precalc->PRE_b0_mode;

            // 李代数到xy的导数
            Vec6f d_xi_x, d_xi_y;

            // Calib到xy的导数
            Vec4f d_C_x, d_C_y;

            // xy 到 idepth 的导数
            float d_d_x, d_d_y;

            float drescale, u, v, new_idepth;  // data in target
            // NOTE u = X/Z, v=Y/Z in target
            float Ku, Kv;
            Vec3f KliP;

            // 重投影
            shared_ptr<PointHessian> p = point.lock();
            if (!projectPoint(p->u, p->v, p->idepth_zero_scaled, 0, 0, HCalib,
                              PRE_RTll_0, PRE_tTll_0, drescale, u, v, Ku, Kv, KliP, new_idepth)) {
                state_NewState = ResState::OOB;
                return state_energy;
            }

            // 各种导数
            // diff d_idepth
            d_d_x = drescale * (PRE_tTll_0[0] - PRE_tTll_0[2] * u) * SCALE_IDEPTH * HCalib->fxl();
            d_d_y = drescale * (PRE_tTll_0[1] - PRE_tTll_0[2] * v) * SCALE_IDEPTH * HCalib->fyl();

            // diff calib
            d_C_x[2] = drescale * (PRE_RTll_0(2, 0) * u - PRE_RTll_0(0, 0));
            d_C_x[3] = HCalib->fxl() * drescale * (PRE_RTll_0(2, 1) * u - PRE_RTll_0(0, 1)) * HCalib->fyli();
            d_C_x[0] = KliP[0] * d_C_x[2];
            d_C_x[1] = KliP[1] * d_C_x[3];

            d_C_y[2] = HCalib->fyl() * drescale * (PRE_RTll_0(2, 0) * v - PRE_RTll_0(1, 0)) * HCalib->fxli();
            d_C_y[3] = drescale * (PRE_RTll_0(2, 1) * v - PRE_RTll_0(1, 1));
            d_C_y[0] = KliP[0] * d_C_y[2];
            d_C_y[1] = KliP[1] * d_C_y[3];

            d_C_x[0] = (d_C_x[0] + u) * SCALE_F;
            d_C_x[1] *= SCALE_F;
            d_C_x[2] = (d_C_x[2] + 1) * SCALE_C;
            d_C_x[3] *= SCALE_C;

            d_C_y[0] *= SCALE_F;
            d_C_y[1] = (d_C_y[1] + v) * SCALE_F;
            d_C_y[2] *= SCALE_C;
            d_C_y[3] = (d_C_y[3] + 1) * SCALE_C;

            // xy到李代数的导数，形式见十四讲
            d_xi_x[0] = new_idepth * HCalib->fxl();
            d_xi_x[1] = 0;
            d_xi_x[2] = -new_idepth * u * HCalib->fxl();
            d_xi_x[3] = -u * v * HCalib->fxl();
            d_xi_x[4] = (1 + u * u) * HCalib->fxl();
            d_xi_x[5] = -v * HCalib->fxl();

            d_xi_y[0] = 0;
            d_xi_y[1] = new_idepth * HCalib->fyl();
            d_xi_y[2] = -new_idepth * v * HCalib->fyl();
            d_xi_y[3] = -(1 + v * v) * HCalib->fyl();
            d_xi_y[4] = u * v * HCalib->fyl();
            d_xi_y[5] = u * HCalib->fyl();


            J->Jpdxi[0] = d_xi_x;
            J->Jpdxi[1] = d_xi_y;

            J->Jpdc[0] = d_C_x;
            J->Jpdc[1] = d_C_y;

            J->Jpdd[0] = d_d_x;
            J->Jpdd[1] = d_d_y;

            Vec2f residual = Vec2f(Ku, Kv) - obsPixel;
            // LOG(INFO) << "proj: " << Ku << ", " << Kv << ", obs: " << obsPixel[0] << ", " << obsPixel[1] << ", res="
            // << residual.transpose() << endl;
            float residualNorm = residual.squaredNorm();
            float w = 1;

            // huber weight
            float hw = fabsf(residualNorm) < setting_huberTH ? 1 : setting_huberTH / fabsf(residualNorm);
            energyLeft += w * w * hw * residual.dot(residual) * (2 - hw);

            if (hw < 1) hw = sqrtf(hw);
            hw = hw * w;
            J->resF = hw * residual;

            state_NewEnergyWithOutlier = energyLeft;

            if (energyLeft > std::max<float>(f->frameEnergyTH, ftarget->frameEnergyTH)) {
                energyLeft = std::max<float>(f->frameEnergyTH, ftarget->frameEnergyTH);
                state_NewState = ResState::OUTLIER;
            } else {
                state_NewState = ResState::IN;
            }

            // LOG(INFO) << "Energy = " << energyLeft << endl;
            state_NewEnergy = energyLeft;
            return energyLeft;
        }

        void FeatureObsResidual::fixLinearizationF(shared_ptr<EnergyFunctional> ef) {

            Vec8f dp = ef->adHTdeltaF[hostIDX + ef->nFrames * targetIDX];

            // compute Jp*delta
            float Jp_delta_x = (J->Jpdxi[0].dot(dp.head<6>())
                                + J->Jpdc[0].dot(ef->cDeltaF)
                                + J->Jpdd[0] * point.lock()->deltaF);
            float Jp_delta_y = (J->Jpdxi[1].dot(dp.head<6>())
                                + J->Jpdc[1].dot(ef->cDeltaF)
                                + J->Jpdd[1] * point.lock()->deltaF);

            res_toZeroF[0] = Jp_delta_x + J->resF[0];
            res_toZeroF[1] = Jp_delta_y + J->resF[1];
            isLinearized = true;
        }
         */

    }

}