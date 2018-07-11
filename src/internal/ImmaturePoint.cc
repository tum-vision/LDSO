#include "Frame.h"
#include "Settings.h"

#include "internal/GlobalCalib.h"
#include "internal/ImmaturePoint.h"
#include "internal/GlobalFuncs.h"
#include "internal/FrameHessian.h"
#include "internal/ResidualProjections.h"

namespace ldso {

    namespace internal {

        ImmaturePoint::ImmaturePoint(shared_ptr<Frame> hostFrame, shared_ptr<Feature> hostFeat, float type,
                                     shared_ptr<CalibHessian> &HCalib) :
                my_type(type), feature(hostFeat) {
            assert(hostFrame->frameHessian);
            gradH.setZero();
            shared_ptr<FrameHessian> host = hostFrame->frameHessian;
            float u = feature->uv[0], v = feature->uv[1];
            for (int idx = 0; idx < patternNum; idx++) {
                int dx = patternP[idx][0];
                int dy = patternP[idx][1];

                Vec3f ptc = getInterpolatedElement33BiLin(host->dI, u + dx, v + dy, wG[0]);

                color[idx] = ptc[0];
                if (!std::isfinite(color[idx])) {
                    energyTH = NAN;
                    return;
                }

                gradH += ptc.tail<2>() * ptc.tail<2>().transpose();
                weights[idx] = sqrtf(
                        setting_outlierTHSumComponent / (setting_outlierTHSumComponent + ptc.tail<2>().squaredNorm()));
            }
            energyTH = patternNum * setting_outlierTH;
            energyTH *= setting_overallEnergyTHWeight * setting_overallEnergyTHWeight;
        }

        /*
         * returns
         * * OOB -> point is optimized and marginalized
         * * UPDATED -> point has been updated.
         * * SKIP -> point has not been updated.
         */
        ImmaturePointStatus ImmaturePoint::traceOn(
                shared_ptr<FrameHessian> frame, const Mat33f &hostToFrame_KRKi,
                const Vec3f &hostToFrame_Kt, const Vec2f &hostToFrame_affine,
                shared_ptr<CalibHessian> HCalib) {

            if (lastTraceStatus == ImmaturePointStatus::IPS_OOB) return lastTraceStatus;
            float maxPixSearch = (wG[0] + hG[0]) * setting_maxPixSearch;

            // ============== project min and max. return if one of them is OOB ===================
            // step 1. 检查极线上点的位置
            // check idepthmin, 最近距离
            Vec3f pr = hostToFrame_KRKi * Vec3f(feature->uv[0], feature->uv[1], 1);
            Vec3f ptpMin = pr + hostToFrame_Kt * idepth_min;
            float uMin = ptpMin[0] / ptpMin[2];
            float vMin = ptpMin[1] / ptpMin[2];

            if (!(uMin > 4 && vMin > 4 && uMin < wG[0] - 5 && vMin < hG[0] - 5)) {
                // out of boundary
                lastTraceUV = Vec2f(-1, -1);
                lastTracePixelInterval = 0;
                return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
            }

            // check idepthmax, maybe infinite
            // 最远距离（注意可能是无限远）
            float dist;
            float uMax;
            float vMax;
            Vec3f ptpMax;   // 按照最远距离来算，在当前帧的投影

            if (std::isfinite(idepth_max)) {
                // 有限远，finite max depth
                ptpMax = pr + hostToFrame_Kt * idepth_max;
                uMax = ptpMax[0] / ptpMax[2];
                vMax = ptpMax[1] / ptpMax[2];

                if (!(uMax > 4 && vMax > 4 && uMax < wG[0] - 5 && vMax < hG[0] - 5)) {
                    lastTraceUV = Vec2f(-1, -1);
                    lastTracePixelInterval = 0;
                    return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
                }

                // ============== check their distance. everything below 2px is OK (-> skip). ===================
                dist = (uMin - uMax) * (uMin - uMax) + (vMin - vMax) * (vMin - vMax);
                dist = sqrtf(dist);
                if (dist < setting_trace_slackInterval /* =2 by default */ ) {
                    // 极线上两个像素非常接近
                    lastTraceUV = Vec2f(uMax + uMin, vMax + vMin) * 0.5;
                    lastTracePixelInterval = dist;
                    return lastTraceStatus = ImmaturePointStatus::IPS_SKIPPED;
                }
                assert(dist > 0);
            } else {
                dist = maxPixSearch;

                // 任取一个距离，idepth=0.01, so depth=100
                // project to arbitrary depth to get direction.
                ptpMax = pr + hostToFrame_Kt * 0.01;
                uMax = ptpMax[0] / ptpMax[2];
                vMax = ptpMax[1] / ptpMax[2];

                // direction.
                float dx = uMax - uMin;
                float dy = vMax - vMin;
                float d = 1.0f / sqrtf(dx * dx + dy * dy);

                // set to [setting_maxPixSearch].
                uMax = uMin + dist * dx * d;
                vMax = vMin + dist * dy * d;

                // may still be out!
                if (!(uMax > 4 && vMax > 4 && uMax < wG[0] - 5 && vMax < hG[0] - 5)) {
                    lastTraceUV = Vec2f(-1, -1);
                    lastTracePixelInterval = 0;
                    return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
                }
                assert(dist > 0);
            }

            // set OOB if scale change too big.
            if (!(idepth_min < 0 || (ptpMin[2] > 0.75 && ptpMin[2] < 1.5))) {
                lastTraceUV = Vec2f(-1, -1);
                lastTracePixelInterval = 0;
                return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
            }

            // ============== compute error-bounds on result in pixel. if the new interval is not at least 1/2 of the old, SKIP ===================
            float dx = setting_trace_stepsize * (uMax - uMin);
            float dy = setting_trace_stepsize * (vMax - vMin);

            float a = (Vec2f(dx, dy).transpose() * gradH * Vec2f(dx, dy));
            float b = (Vec2f(dy, -dx).transpose() * gradH * Vec2f(dy, -dx));
            float errorInPixel = 0.2f + 0.2f * (a + b) / a;

            if (errorInPixel * setting_trace_minImprovementFactor > dist && std::isfinite(idepth_max)) {
                lastTraceUV = Vec2f(uMax + uMin, vMax + vMin) * 0.5;
                lastTracePixelInterval = dist;
                return lastTraceStatus = ImmaturePointStatus::IPS_BADCONDITION;
            }

            if (errorInPixel > 10) errorInPixel = 10;

            // ============== do the discrete search ===================
            dx /= dist;
            dy /= dist;

            if (dist > maxPixSearch) {
                uMax = uMin + maxPixSearch * dx;
                vMax = vMin + maxPixSearch * dy;
                dist = maxPixSearch;
            }

            int numSteps = 1.9999f + dist / setting_trace_stepsize;
            Mat22f Rplane = hostToFrame_KRKi.topLeftCorner<2, 2>();

            float randShift = uMin * 1000 - floorf(uMin * 1000);
            float ptx = uMin - randShift * dx;
            float pty = vMin - randShift * dy;


            Vec2f rotatetPattern[MAX_RES_PER_POINT];
            for (int idx = 0; idx < patternNum; idx++)
                rotatetPattern[idx] = Rplane * Vec2f(patternP[idx][0], patternP[idx][1]);

            if (!std::isfinite(dx) || !std::isfinite(dy)) {
                lastTracePixelInterval = 0;
                lastTraceUV = Vec2f(-1, -1);
                return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
            }

            float errors[100];
            float bestU = 0, bestV = 0, bestEnergy = 1e10;
            int bestIdx = -1;
            if (numSteps >= 100) numSteps = 99;

            for (int i = 0; i < numSteps; i++) {
                float energy = 0;
                for (int idx = 0; idx < patternNum; idx++) {
                    float hitColor = getInterpolatedElement31(frame->dI,
                                                              (float) (ptx + rotatetPattern[idx][0]),
                                                              (float) (pty + rotatetPattern[idx][1]),
                                                              wG[0]);

                    if (!std::isfinite(hitColor)) {
                        energy += 1e5;
                        continue;
                    }
                    float residual = hitColor - (float) (hostToFrame_affine[0] * color[idx] + hostToFrame_affine[1]);
                    float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);
                    energy += hw * residual * residual * (2 - hw);
                }

                errors[i] = energy;
                if (energy < bestEnergy) {
                    bestU = ptx;
                    bestV = pty;
                    bestEnergy = energy;
                    bestIdx = i;
                }

                ptx += dx;
                pty += dy;
            }


            // find best score outside a +-2px radius.
            float secondBest = 1e10;
            for (int i = 0; i < numSteps; i++) {
                if ((i < bestIdx - setting_minTraceTestRadius || i > bestIdx + setting_minTraceTestRadius) &&
                    errors[i] < secondBest)
                    secondBest = errors[i];
            }
            float newQuality = secondBest / bestEnergy;
            if (newQuality < quality || numSteps > 10) quality = newQuality;


            // ============== do GN optimization ===================
            float uBak = bestU, vBak = bestV, gnstepsize = 1, stepBack = 0;
            if (setting_trace_GNIterations > 0) bestEnergy = 1e5;
            int gnStepsGood = 0, gnStepsBad = 0;
            for (int it = 0; it < setting_trace_GNIterations; it++) {
                float H = 1, b = 0, energy = 0;
                for (int idx = 0; idx < patternNum; idx++) {
                    Vec3f hitColor = getInterpolatedElement33(frame->dI,
                                                              (float) (bestU + rotatetPattern[idx][0]),
                                                              (float) (bestV + rotatetPattern[idx][1]), wG[0]);

                    if (!std::isfinite((float) hitColor[0])) {
                        energy += 1e5;
                        continue;
                    }
                    float residual = hitColor[0] - (hostToFrame_affine[0] * color[idx] + hostToFrame_affine[1]);
                    float dResdDist = dx * hitColor[1] + dy * hitColor[2];
                    float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);

                    H += hw * dResdDist * dResdDist;
                    b += hw * residual * dResdDist;
                    energy += weights[idx] * weights[idx] * hw * residual * residual * (2 - hw);
                }


                if (energy > bestEnergy) {
                    gnStepsBad++;

                    // do a smaller step from old point.
                    stepBack *= 0.5;
                    bestU = uBak + stepBack * dx;
                    bestV = vBak + stepBack * dy;
                } else {
                    gnStepsGood++;

                    float step = -gnstepsize * b / H;
                    if (step < -0.5) step = -0.5;
                    else if (step > 0.5) step = 0.5;

                    if (!std::isfinite(step)) step = 0;

                    uBak = bestU;
                    vBak = bestV;
                    stepBack = step;

                    bestU += step * dx;
                    bestV += step * dy;
                    bestEnergy = energy;
                }

                if (fabsf(stepBack) < setting_trace_GNThreshold) break;
            }

            // ============== detect energy-based outlier. ===================
            if (!(bestEnergy < energyTH * setting_trace_extraSlackOnTH)) {
                lastTracePixelInterval = 0;
                lastTraceUV = Vec2f(-1, -1);
                if (lastTraceStatus == ImmaturePointStatus::IPS_OUTLIER)
                    return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
                else
                    return lastTraceStatus = ImmaturePointStatus::IPS_OUTLIER;
            }

            // ============== set new interval ===================
            if (dx * dx > dy * dy) {
                idepth_min = (pr[2] * (bestU - errorInPixel * dx) - pr[0]) /
                             (hostToFrame_Kt[0] - hostToFrame_Kt[2] * (bestU - errorInPixel * dx));
                idepth_max = (pr[2] * (bestU + errorInPixel * dx) - pr[0]) /
                             (hostToFrame_Kt[0] - hostToFrame_Kt[2] * (bestU + errorInPixel * dx));
            } else {
                idepth_min = (pr[2] * (bestV - errorInPixel * dy) - pr[1]) /
                             (hostToFrame_Kt[1] - hostToFrame_Kt[2] * (bestV - errorInPixel * dy));
                idepth_max = (pr[2] * (bestV + errorInPixel * dy) - pr[1]) /
                             (hostToFrame_Kt[1] - hostToFrame_Kt[2] * (bestV + errorInPixel * dy));
            }
            if (idepth_min > idepth_max) std::swap<float>(idepth_min, idepth_max);


            if (!std::isfinite(idepth_min) || !std::isfinite(idepth_max) || (idepth_max < 0)) {
                lastTracePixelInterval = 0;
                lastTraceUV = Vec2f(-1, -1);
                return lastTraceStatus = ImmaturePointStatus::IPS_OUTLIER;
            }

            lastTracePixelInterval = 2 * errorInPixel;
            lastTraceUV = Vec2f(bestU, bestV);
            return lastTraceStatus = ImmaturePointStatus::IPS_GOOD;
        }

        double ImmaturePoint::linearizeResidual(
                shared_ptr<CalibHessian> HCalib, const float outlierTHSlack,
                shared_ptr<ImmaturePointTemporaryResidual> tmpRes, float &Hdd, float &bd,
                float idepth) {

            if (tmpRes->state_state == ResState::OOB) {
                tmpRes->state_NewState = ResState::OOB;
                return tmpRes->state_energy;
            }

            shared_ptr<FrameHessian> host = feature->host.lock()->frameHessian;
            shared_ptr<FrameHessian> target = tmpRes->target.lock();
            FrameFramePrecalc *precalc = &(host->targetPrecalc[target->idx]);

            // check OOB due to scale angle change.
            float energyLeft = 0;
            const Eigen::Vector3f *dIl = target->dI;
            const Mat33f &PRE_RTll = precalc->PRE_RTll;
            const Vec3f &PRE_tTll = precalc->PRE_tTll;

            Vec2f affLL = precalc->PRE_aff_mode;

            for (int idx = 0; idx < patternNum; idx++) {
                int dx = patternP[idx][0];
                int dy = patternP[idx][1];

                float drescale, u, v, new_idepth;
                float Ku, Kv;
                Vec3f KliP;

                if (!projectPoint(this->feature->uv[0], this->feature->uv[1], idepth, dx, dy, HCalib,
                                  PRE_RTll, PRE_tTll, drescale, u, v, Ku, Kv, KliP, new_idepth)) {
                    tmpRes->state_NewState = ResState::OOB;
                    return tmpRes->state_energy;
                }


                Vec3f hitColor = (getInterpolatedElement33(dIl, Ku, Kv, wG[0]));

                if (!std::isfinite((float) hitColor[0])) {
                    tmpRes->state_NewState = ResState::OOB;
                    return tmpRes->state_energy;
                }
                float residual = hitColor[0] - (affLL[0] * color[idx] + affLL[1]);

                float hw = fabsf(residual) < setting_huberTH ? 1 : setting_huberTH / fabsf(residual);
                energyLeft += weights[idx] * weights[idx] * hw * residual * residual * (2 - hw);

                // depth derivatives.
                float dxInterp = hitColor[1] * HCalib->fxl();
                float dyInterp = hitColor[2] * HCalib->fyl();
                float d_idepth = derive_idepth(PRE_tTll, u, v, dx, dy, dxInterp, dyInterp, drescale);

                hw *= weights[idx] * weights[idx];

                Hdd += (hw * d_idepth) * d_idepth;
                bd += (hw * residual) * d_idepth;
            }


            if (energyLeft > energyTH * outlierTHSlack) {
                energyLeft = energyTH * outlierTHSlack;
                tmpRes->state_NewState = ResState::OUTLIER;
            } else {
                tmpRes->state_NewState = ResState::IN;
            }

            tmpRes->state_NewEnergy = energyLeft;
            return energyLeft;
        }

        float ImmaturePoint::calcResidual(
                shared_ptr<CalibHessian> HCalib, const float outlierTHSlack,
                shared_ptr<ImmaturePointTemporaryResidual> tmpRes, float idepth) {
            shared_ptr<FrameHessian> host = feature->host.lock()->frameHessian;
            shared_ptr<FrameHessian> target = tmpRes->target.lock();
            FrameFramePrecalc *precalc = &(host->targetPrecalc[target->idx]);
            float energyLeft = 0;
            const Eigen::Vector3f *dIl = target->dI;
            const Mat33f &PRE_KRKiTll = precalc->PRE_KRKiTll;
            const Vec3f &PRE_KtTll = precalc->PRE_KtTll;
            Vec2f affLL = precalc->PRE_aff_mode;

            for (int idx = 0; idx < patternNum; idx++) {
                float Ku, Kv;
                if (!projectPoint(this->feature->uv[0] + patternP[idx][0], this->feature->uv[1] + patternP[idx][1],
                                  idepth, PRE_KRKiTll, PRE_KtTll, Ku, Kv)) { return 1e10; }

                Vec3f hitColor = (getInterpolatedElement33(dIl, Ku, Kv, wG[0]));
                if (!std::isfinite((float) hitColor[0])) {
                    return 1e10;
                }

                float residual = hitColor[0] - (affLL[0] * color[idx] + affLL[1]);

                float hw = fabsf(residual) < setting_huberTH ? 1 : setting_huberTH / fabsf(residual);
                energyLeft += weights[idx] * weights[idx] * hw * residual * residual * (2 - hw);
            }

            if (energyLeft > energyTH * outlierTHSlack) {
                energyLeft = energyTH * outlierTHSlack;
            }
            return energyLeft;
        }

        float ImmaturePoint::getdPixdd(
                shared_ptr<CalibHessian> HCalib,
                shared_ptr<ImmaturePointTemporaryResidual> tmpRes, float idepth) {

            shared_ptr<FrameHessian> host = feature->host.lock()->frameHessian;
            shared_ptr<FrameHessian> target = tmpRes->target.lock();

            FrameFramePrecalc *precalc = &(host->targetPrecalc[target->idx]);
            const Vec3f &PRE_tTll = precalc->PRE_tTll;
            float drescale, u = 0, v = 0, new_idepth;
            float Ku, Kv;
            Vec3f KliP;

            projectPoint(this->feature->uv[0], this->feature->uv[1], idepth, 0, 0, HCalib,
                         precalc->PRE_RTll, PRE_tTll, drescale, u, v, Ku, Kv, KliP, new_idepth);

            float dxdd = (PRE_tTll[0] - PRE_tTll[2] * u) * HCalib->fxl();
            float dydd = (PRE_tTll[1] - PRE_tTll[2] * v) * HCalib->fyl();
            return drescale * sqrtf(dxdd * dxdd + dydd * dydd);
        }

    }
}