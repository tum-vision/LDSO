#include "Feature.h"
#include "Point.h"

#include "frontend/CoarseTracker.h"
#include "internal/GlobalCalib.h"
#include "internal/FrameHessian.h"
#include "internal/PointHessian.h"
#include "internal/CalibHessian.h"
#include "internal/GlobalFuncs.h"

namespace ldso {

    /**
     * Create an aligned array and send their pointer into rawPtrVec
     * @tparam b bit of each element
     * @tparam T type of allocated buffer
     * @param size size of the buffer
     * @param rawPtrVec allocated pointer buffer, will be deleted after
     * @return allocated pointer
     */
    template<int b, typename T>
    T *allocAligned(int size, std::vector<T *> &rawPtrVec) {
        const int padT = 1 + ((1 << b) / sizeof(T));
        T *ptr = new T[size + padT];
        rawPtrVec.push_back(ptr);
        T *alignedPtr = (T *) ((((uintptr_t) (ptr + padT)) >> b) << b);
        return alignedPtr;
    }

    CoarseTracker::CoarseTracker(int ww, int hh) {

        // make coarse tracking templates.
        for (int lvl = 0; lvl < pyrLevelsUsed; lvl++) {
            int wl = ww >> lvl;
            int hl = hh >> lvl;

            idepth[lvl] = allocAligned<4, float>(wl * hl, ptrToDelete);
            weightSums[lvl] = allocAligned<4, float>(wl * hl, ptrToDelete);
            weightSums_bak[lvl] = allocAligned<4, float>(wl * hl, ptrToDelete);

            pc_u[lvl] = allocAligned<4, float>(wl * hl, ptrToDelete);
            pc_v[lvl] = allocAligned<4, float>(wl * hl, ptrToDelete);
            pc_idepth[lvl] = allocAligned<4, float>(wl * hl, ptrToDelete);
            pc_color[lvl] = allocAligned<4, float>(wl * hl, ptrToDelete);

        }

        // warped buffers
        buf_warped_idepth = allocAligned<4, float>(ww * hh, ptrToDelete);
        buf_warped_u = allocAligned<4, float>(ww * hh, ptrToDelete);
        buf_warped_v = allocAligned<4, float>(ww * hh, ptrToDelete);
        buf_warped_dx = allocAligned<4, float>(ww * hh, ptrToDelete);
        buf_warped_dy = allocAligned<4, float>(ww * hh, ptrToDelete);
        buf_warped_residual = allocAligned<4, float>(ww * hh, ptrToDelete);
        buf_warped_weight = allocAligned<4, float>(ww * hh, ptrToDelete);
        buf_warped_refColor = allocAligned<4, float>(ww * hh, ptrToDelete);

        w[0] = h[0] = 0;
    }

    bool CoarseTracker::trackNewestCoarse(
            shared_ptr<FrameHessian> newFrameHessian, SE3 &lastToNew_out,
            AffLight &aff_g2l_out, int coarsestLvl, Vec5 minResForAbort) {

        assert(coarsestLvl < 5 && coarsestLvl < pyrLevelsUsed);

        lastResiduals.setConstant(NAN);
        lastFlowIndicators.setConstant(1000);

        newFrame = newFrameHessian;

        // iteration in each pyramid level
        int maxIterations[] = {10, 20, 50, 50, 50};
        float lambdaExtrapolationLimit = 0.001;

        // use last track results as an initial guess
        SE3 refToNew_current = lastToNew_out;
        AffLight aff_g2l_current = aff_g2l_out;

        bool haveRepeated = false;

        // coarse-to-fine
        for (int lvl = coarsestLvl; lvl >= 0; lvl--) {
            Mat88 H;
            Vec8 b;
            float levelCutoffRepeat = 1;

            // compute the residual and adjust the huber threshold
            Vec6 resOld = calcRes(lvl, refToNew_current, aff_g2l_current, setting_coarseCutoffTH * levelCutoffRepeat);
            while (resOld[5] > 0.6 && levelCutoffRepeat < 50) {
                // more than 60% is over than threshold, then increate the cut off threshold
                levelCutoffRepeat *= 2;
                resOld = calcRes(lvl, refToNew_current, aff_g2l_current, setting_coarseCutoffTH * levelCutoffRepeat);
            }

            // Compute H and b
            calcGSSSE(lvl, H, b, refToNew_current, aff_g2l_current);

            float lambda = 0.01;

            // relAff no used?
            //Vec2f relAff = AffLight::fromToVecExposure(lastRef->ab_exposure, newFrame->ab_exposure, lastRef_aff_g2l,
            //                                           aff_g2l_current).cast<float>();

            // L-M iteration
            for (int iteration = 0; iteration < maxIterations[lvl]; iteration++) {
                Mat88 Hl = H;
                for (int i = 0; i < 8; i++) Hl(i, i) *= (1 + lambda);
                Vec8 inc = Hl.ldlt().solve(-b);

                // depends on the mode, if a,b is fixed, don't estimate them
                if (setting_affineOptModeA < 0 && setting_affineOptModeB < 0)    // fix a, b
                {
                    inc.head<6>() = Hl.topLeftCorner<6, 6>().ldlt().solve(-b.head<6>());
                    inc.tail<2>().setZero();
                }
                if (!(setting_affineOptModeA < 0) && setting_affineOptModeB < 0)    // fix b
                {
                    inc.head<7>() = Hl.topLeftCorner<7, 7>().ldlt().solve(-b.head<7>());
                    inc.tail<1>().setZero();
                }
                if (setting_affineOptModeA < 0 && !(setting_affineOptModeB < 0))    // fix a
                {
                    Mat88 HlStitch = Hl;
                    Vec8 bStitch = b;
                    HlStitch.col(6) = HlStitch.col(7);
                    HlStitch.row(6) = HlStitch.row(7);
                    bStitch[6] = bStitch[7];
                    Vec7 incStitch = HlStitch.topLeftCorner<7, 7>().ldlt().solve(-bStitch.head<7>());
                    inc.setZero();
                    inc.head<6>() = incStitch.head<6>();
                    inc[6] = 0;
                    inc[7] = incStitch[6];
                }

                float extrapFac = 1;
                if (lambda < lambdaExtrapolationLimit)
                    extrapFac = sqrtf(sqrt(lambdaExtrapolationLimit / lambda));
                inc *= extrapFac;

                Vec8 incScaled = inc;
                incScaled.segment<3>(0) *= SCALE_XI_ROT;
                incScaled.segment<3>(3) *= SCALE_XI_TRANS;
                incScaled.segment<1>(6) *= SCALE_A;
                incScaled.segment<1>(7) *= SCALE_B;

                if (!std::isfinite(incScaled.sum())) incScaled.setZero();

                // left multiply the pose and add to a,b
                SE3 refToNew_new = SE3::exp((Vec6) (incScaled.head<6>())) * refToNew_current;
                AffLight aff_g2l_new = aff_g2l_current;
                aff_g2l_new.a += incScaled[6];
                aff_g2l_new.b += incScaled[7];

                // calculate new residual after this update step
                Vec6 resNew = calcRes(lvl, refToNew_new, aff_g2l_new, setting_coarseCutoffTH * levelCutoffRepeat);

                // decide whether to accept this step
                // res[0]/res[1] is the average energy
                bool accept = (resNew[0] / resNew[1]) < (resOld[0] / resOld[1]);

                // relAff no used?
                //Vec2f relAff = AffLight::fromToVecExposure(lastRef->ab_exposure, newFrame->ab_exposure,
                //                                           lastRef_aff_g2l, aff_g2l_new).cast<float>();
                if (accept) {

                    // decrease lambda
                    calcGSSSE(lvl, H, b, refToNew_new, aff_g2l_new);
                    resOld = resNew;
                    aff_g2l_current = aff_g2l_new;
                    refToNew_current = refToNew_new;
                    lambda *= 0.5;
                } else {
                    // increase lambda in LM
                    lambda *= 4;
                    if (lambda < lambdaExtrapolationLimit) lambda = lambdaExtrapolationLimit;
                }

                // terminate if increment is small
                if (!(inc.norm() > 1e-3)) {
                    break;
                }
            } // end of L-M iteration

            // set last residual for that level, as well as flow indicators.
            lastResiduals[lvl] = sqrtf((float) (resOld[0] / resOld[1]));
            lastFlowIndicators = resOld.segment<3>(2);
            if (lastResiduals[lvl] > 1.5 * minResForAbort[lvl])
                return false;

            // repeat this level level
            if (levelCutoffRepeat > 1 && !haveRepeated) {
                lvl++;
                haveRepeated = true;
            }
        } // end of for: pyramid level

        // set!
        lastToNew_out = refToNew_current;
        aff_g2l_out = aff_g2l_current;

        if ((setting_affineOptModeA != 0 && (fabsf(aff_g2l_out.a) > 1.2))
            || (setting_affineOptModeB != 0 && (fabsf(aff_g2l_out.b) > 200)))
            return false;

        Vec2f relAff = AffLight::fromToVecExposure(lastRef->ab_exposure, newFrame->ab_exposure, lastRef_aff_g2l,
                                                   aff_g2l_out).cast<float>();

        if ((setting_affineOptModeA == 0 && (fabsf(logf((float) relAff[0])) > 1.5))
            || (setting_affineOptModeB == 0 && (fabsf((float) relAff[1]) > 200)))
            return false;

        if (setting_affineOptModeA < 0) aff_g2l_out.a = 0;
        if (setting_affineOptModeB < 0) aff_g2l_out.b = 0;

        return true;
    }

    void CoarseTracker::makeK(shared_ptr<CalibHessian> HCalib) {

        w[0] = wG[0];
        h[0] = hG[0];

        fx[0] = HCalib->fxl();
        fy[0] = HCalib->fyl();
        cx[0] = HCalib->cxl();
        cy[0] = HCalib->cyl();

        for (int level = 1; level < pyrLevelsUsed; ++level) {
            w[level] = w[0] >> level;
            h[level] = h[0] >> level;
            fx[level] = fx[level - 1] * 0.5;
            fy[level] = fy[level - 1] * 0.5;
            cx[level] = (cx[0] + 0.5) / ((int) 1 << level) - 0.5;
            cy[level] = (cy[0] + 0.5) / ((int) 1 << level) - 0.5;
        }

        for (int level = 0; level < pyrLevelsUsed; ++level) {
            K[level] << fx[level], 0.0, cx[level], 0.0, fy[level], cy[level], 0.0, 0.0, 1.0;
            Ki[level] = K[level].inverse();
            fxi[level] = Ki[level](0, 0);
            fyi[level] = Ki[level](1, 1);
            cxi[level] = Ki[level](0, 2);
            cyi[level] = Ki[level](1, 2);
        }
    }

    void CoarseTracker::setCoarseTrackingRef(std::vector<shared_ptr<FrameHessian>> &frameHessians) {

        assert(frameHessians.size() > 0);
        lastRef = frameHessians.back();
        makeCoarseDepthL0(frameHessians);
        refFrameID = lastRef->frame->id;
        lastRef_aff_g2l = lastRef->aff_g2l();
        firstCoarseRMSE = -1;
    }

    void CoarseTracker::makeCoarseDepthL0(std::vector<shared_ptr<FrameHessian>> frameHessians) {

        // make coarse tracking templates for latstRef.
        memset(idepth[0], 0, sizeof(float) * w[0] * h[0]);
        memset(weightSums[0], 0, sizeof(float) * w[0] * h[0]);

        for (shared_ptr<FrameHessian> fh: frameHessians) {
            for (shared_ptr<Feature> feat: fh->frame->features) {
                if (feat->status == Feature::FeatureStatus::VALID &&
                    feat->point->status == Point::PointStatus::ACTIVE) {

                    shared_ptr<PointHessian> ph = feat->point->mpPH;
                    if (ph->lastResiduals[0].first != 0 && ph->lastResiduals[0].second == ResState::IN) {
                        shared_ptr<PointFrameResidual> r = ph->lastResiduals[0].first;
                        assert(r->isActive() && r->target.lock() == lastRef);
                        int u = r->centerProjectedTo[0] + 0.5f;
                        int v = r->centerProjectedTo[1] + 0.5f;
                        float new_idepth = r->centerProjectedTo[2];
                        float weight = sqrtf(1e-3 / (ph->HdiF + 1e-12));

                        idepth[0][u + w[0] * v] += new_idepth * weight;
                        weightSums[0][u + w[0] * v] += weight;
                    }
                }
            }
        }

        for (int lvl = 1; lvl < pyrLevelsUsed; lvl++) {
            int lvlm1 = lvl - 1;
            int wl = w[lvl], hl = h[lvl], wlm1 = w[lvlm1];

            float *idepth_l = idepth[lvl];
            float *weightSums_l = weightSums[lvl];

            float *idepth_lm = idepth[lvlm1];
            float *weightSums_lm = weightSums[lvlm1];

            for (int y = 0; y < hl; y++)
                for (int x = 0; x < wl; x++) {
                    int bidx = 2 * x + 2 * y * wlm1;
                    idepth_l[x + y * wl] =
                            idepth_lm[bidx] +
                            idepth_lm[bidx + 1] +
                            idepth_lm[bidx + wlm1] +
                            idepth_lm[bidx + wlm1 + 1];

                    weightSums_l[x + y * wl] =
                            weightSums_lm[bidx] +
                            weightSums_lm[bidx + 1] +
                            weightSums_lm[bidx + wlm1] +
                            weightSums_lm[bidx + wlm1 + 1];
                }
        }

        // dilate idepth by 1.
        for (int lvl = 0; lvl < 2; lvl++) {
            int numIts = 1;


            for (int it = 0; it < numIts; it++) {
                int wh = w[lvl] * h[lvl] - w[lvl];
                int wl = w[lvl];
                float *weightSumsl = weightSums[lvl];
                float *weightSumsl_bak = weightSums_bak[lvl];
                memcpy(weightSumsl_bak, weightSumsl, w[lvl] * h[lvl] * sizeof(float));
                float *idepthl = idepth[lvl];    // dotnt need to make a temp copy of depth, since I only
                // read values with weightSumsl>0, and write ones with weightSumsl<=0.
                for (int i = w[lvl]; i < wh; i++) {
                    if (weightSumsl_bak[i] <= 0) {
                        float sum = 0, num = 0, numn = 0;
                        if (weightSumsl_bak[i + 1 + wl] > 0) {
                            sum += idepthl[i + 1 + wl];
                            num += weightSumsl_bak[i + 1 + wl];
                            numn++;
                        }
                        if (weightSumsl_bak[i - 1 - wl] > 0) {
                            sum += idepthl[i - 1 - wl];
                            num += weightSumsl_bak[i - 1 - wl];
                            numn++;
                        }
                        if (weightSumsl_bak[i + wl - 1] > 0) {
                            sum += idepthl[i + wl - 1];
                            num += weightSumsl_bak[i + wl - 1];
                            numn++;
                        }
                        if (weightSumsl_bak[i - wl + 1] > 0) {
                            sum += idepthl[i - wl + 1];
                            num += weightSumsl_bak[i - wl + 1];
                            numn++;
                        }
                        if (numn > 0) {
                            idepthl[i] = sum / numn;
                            weightSumsl[i] = num / numn;
                        }
                    }
                }
            }
        }


        // dilate idepth by 1 (2 on lower levels).
        for (int lvl = 2; lvl < pyrLevelsUsed; lvl++) {
            int wh = w[lvl] * h[lvl] - w[lvl];
            int wl = w[lvl];
            float *weightSumsl = weightSums[lvl];
            float *weightSumsl_bak = weightSums_bak[lvl];
            memcpy(weightSumsl_bak, weightSumsl, w[lvl] * h[lvl] * sizeof(float));
            float *idepthl = idepth[lvl];    // dotnt need to make a temp copy of depth, since I only
            // read values with weightSumsl>0, and write ones with weightSumsl<=0.
            for (int i = w[lvl]; i < wh; i++) {
                if (weightSumsl_bak[i] <= 0) {
                    float sum = 0, num = 0, numn = 0;
                    if (weightSumsl_bak[i + 1] > 0) {
                        sum += idepthl[i + 1];
                        num += weightSumsl_bak[i + 1];
                        numn++;
                    }
                    if (weightSumsl_bak[i - 1] > 0) {
                        sum += idepthl[i - 1];
                        num += weightSumsl_bak[i - 1];
                        numn++;
                    }
                    if (weightSumsl_bak[i + wl] > 0) {
                        sum += idepthl[i + wl];
                        num += weightSumsl_bak[i + wl];
                        numn++;
                    }
                    if (weightSumsl_bak[i - wl] > 0) {
                        sum += idepthl[i - wl];
                        num += weightSumsl_bak[i - wl];
                        numn++;
                    }
                    if (numn > 0) {
                        idepthl[i] = sum / numn;
                        weightSumsl[i] = num / numn;
                    }
                }
            }
        }

        // normalize idepths and weights.
        for (int lvl = 0; lvl < pyrLevelsUsed; lvl++) {
            float *weightSumsl = weightSums[lvl];
            float *idepthl = idepth[lvl];
            Eigen::Vector3f *dIRefl = lastRef->dIp[lvl];

            int wl = w[lvl], hl = h[lvl];

            int lpc_n = 0;
            float *lpc_u = pc_u[lvl];
            float *lpc_v = pc_v[lvl];
            float *lpc_idepth = pc_idepth[lvl];
            float *lpc_color = pc_color[lvl];


            for (int y = 2; y < hl - 2; y++)
                for (int x = 2; x < wl - 2; x++) {
                    int i = x + y * wl;

                    if (weightSumsl[i] > 0) {
                        idepthl[i] /= weightSumsl[i];
                        lpc_u[lpc_n] = x;
                        lpc_v[lpc_n] = y;
                        lpc_idepth[lpc_n] = idepthl[i];
                        lpc_color[lpc_n] = dIRefl[i][0];


                        if (!std::isfinite(lpc_color[lpc_n]) || !(idepthl[i] > 0)) {
                            idepthl[i] = -1;
                            continue;    // just skip if something is wrong.
                        }
                        lpc_n++;
                    } else
                        idepthl[i] = -1;

                    weightSumsl[i] = 1;
                }

            pc_n[lvl] = lpc_n;
        }
    }

    Vec6 CoarseTracker::calcRes(int lvl, const SE3 &refToNew, AffLight aff_g2l, float cutoffTH) {

        float E = 0;
        int numTermsInE = 0;
        int numTermsInWarped = 0;
        int numSaturated = 0;

        int wl = w[lvl];
        int hl = h[lvl];
        Eigen::Vector3f *dINewl = newFrame->dIp[lvl];
        float fxl = fx[lvl];
        float fyl = fy[lvl];
        float cxl = cx[lvl];
        float cyl = cy[lvl];


        Mat33f RKi = (refToNew.rotationMatrix().cast<float>() * Ki[lvl]);
        Vec3f t = (refToNew.translation()).cast<float>();
        Vec2f affLL = AffLight::fromToVecExposure(lastRef->ab_exposure, newFrame->ab_exposure, lastRef_aff_g2l,
                                                  aff_g2l).cast<float>();

        float sumSquaredShiftT = 0;
        float sumSquaredShiftRT = 0;
        float sumSquaredShiftNum = 0;

        float maxEnergy = 2 * setting_huberTH * cutoffTH -
                          setting_huberTH * setting_huberTH;    // energy for r=setting_coarseCutoffTH.

        int nl = pc_n[lvl];
        float *lpc_u = pc_u[lvl];
        float *lpc_v = pc_v[lvl];
        float *lpc_idepth = pc_idepth[lvl];
        float *lpc_color = pc_color[lvl];


        for (int i = 0; i < nl; i++) {
            float id = lpc_idepth[i];
            float x = lpc_u[i];
            float y = lpc_v[i];

            Vec3f pt = RKi * Vec3f(x, y, 1) + t * id;
            float u = pt[0] / pt[2];
            float v = pt[1] / pt[2];
            float Ku = fxl * u + cxl;
            float Kv = fyl * v + cyl;
            float new_idepth = id / pt[2];

            if (lvl == 0 && i % 32 == 0) {
                // translation only (positive)
                Vec3f ptT = Ki[lvl] * Vec3f(x, y, 1) + t * id;
                float uT = ptT[0] / ptT[2];
                float vT = ptT[1] / ptT[2];
                float KuT = fxl * uT + cxl;
                float KvT = fyl * vT + cyl;

                // translation only (negative)
                Vec3f ptT2 = Ki[lvl] * Vec3f(x, y, 1) - t * id;
                float uT2 = ptT2[0] / ptT2[2];
                float vT2 = ptT2[1] / ptT2[2];
                float KuT2 = fxl * uT2 + cxl;
                float KvT2 = fyl * vT2 + cyl;

                //translation and rotation (negative)
                Vec3f pt3 = RKi * Vec3f(x, y, 1) - t * id;
                float u3 = pt3[0] / pt3[2];
                float v3 = pt3[1] / pt3[2];
                float Ku3 = fxl * u3 + cxl;
                float Kv3 = fyl * v3 + cyl;

                //translation and rotation (positive)
                //already have it.

                sumSquaredShiftT += (KuT - x) * (KuT - x) + (KvT - y) * (KvT - y);
                sumSquaredShiftT += (KuT2 - x) * (KuT2 - x) + (KvT2 - y) * (KvT2 - y);
                sumSquaredShiftRT += (Ku - x) * (Ku - x) + (Kv - y) * (Kv - y);
                sumSquaredShiftRT += (Ku3 - x) * (Ku3 - x) + (Kv3 - y) * (Kv3 - y);
                sumSquaredShiftNum += 2;
            }

            if (!(Ku > 2 && Kv > 2 && Ku < wl - 3 && Kv < hl - 3 && new_idepth > 0)) continue;


            float refColor = lpc_color[i];
            Vec3f hitColor = getInterpolatedElement33(dINewl, Ku, Kv, wl);
            if (!std::isfinite((float) hitColor[0])) continue;
            float residual = hitColor[0] - (float) (affLL[0] * refColor + affLL[1]);
            float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);


            if (fabs(residual) > cutoffTH) {
                E += maxEnergy;
                numTermsInE++;
                numSaturated++;
            } else {

                E += hw * residual * residual * (2 - hw);
                numTermsInE++;

                buf_warped_idepth[numTermsInWarped] = new_idepth;
                buf_warped_u[numTermsInWarped] = u;
                buf_warped_v[numTermsInWarped] = v;
                buf_warped_dx[numTermsInWarped] = hitColor[1];
                buf_warped_dy[numTermsInWarped] = hitColor[2];
                buf_warped_residual[numTermsInWarped] = residual;
                buf_warped_weight[numTermsInWarped] = hw;
                buf_warped_refColor[numTermsInWarped] = lpc_color[i];
                numTermsInWarped++;
            }
        }

        while (numTermsInWarped % 4 != 0) {
            buf_warped_idepth[numTermsInWarped] = 0;
            buf_warped_u[numTermsInWarped] = 0;
            buf_warped_v[numTermsInWarped] = 0;
            buf_warped_dx[numTermsInWarped] = 0;
            buf_warped_dy[numTermsInWarped] = 0;
            buf_warped_residual[numTermsInWarped] = 0;
            buf_warped_weight[numTermsInWarped] = 0;
            buf_warped_refColor[numTermsInWarped] = 0;
            numTermsInWarped++;
        }
        buf_warped_n = numTermsInWarped;

        Vec6 rs;
        rs[0] = E;
        rs[1] = numTermsInE;
        rs[2] = sumSquaredShiftT / (sumSquaredShiftNum + 0.1);
        rs[3] = 0;
        rs[4] = sumSquaredShiftRT / (sumSquaredShiftNum + 0.1);
        rs[5] = numSaturated / (float) numTermsInE;

        return rs;
    }

    void CoarseTracker::calcGSSSE(int lvl, Mat88 &H_out, Vec8 &b_out, const SE3 &refToNew, AffLight aff_g2l) {

        acc.initialize();

        __m128 fxl = _mm_set1_ps(fx[lvl]);
        __m128 fyl = _mm_set1_ps(fy[lvl]);
        __m128 b0 = _mm_set1_ps(lastRef_aff_g2l.b);
        __m128 a = _mm_set1_ps(
                (float) (AffLight::fromToVecExposure(lastRef->ab_exposure, newFrame->ab_exposure, lastRef_aff_g2l,
                                                     aff_g2l)[0]));

        __m128 one = _mm_set1_ps(1);
        __m128 minusOne = _mm_set1_ps(-1);
        __m128 zero = _mm_set1_ps(0);

        int n = buf_warped_n;
        assert(n % 4 == 0);
        for (int i = 0; i < n; i += 4) {
            __m128 dx = _mm_mul_ps(_mm_load_ps(buf_warped_dx + i), fxl);
            __m128 dy = _mm_mul_ps(_mm_load_ps(buf_warped_dy + i), fyl);
            __m128 u = _mm_load_ps(buf_warped_u + i);
            __m128 v = _mm_load_ps(buf_warped_v + i);
            __m128 id = _mm_load_ps(buf_warped_idepth + i);


            acc.updateSSE_eighted(
                    _mm_mul_ps(id, dx),
                    _mm_mul_ps(id, dy),
                    _mm_sub_ps(zero, _mm_mul_ps(id, _mm_add_ps(_mm_mul_ps(u, dx), _mm_mul_ps(v, dy)))),
                    _mm_sub_ps(zero, _mm_add_ps(
                            _mm_mul_ps(_mm_mul_ps(u, v), dx),
                            _mm_mul_ps(dy, _mm_add_ps(one, _mm_mul_ps(v, v))))),
                    _mm_add_ps(
                            _mm_mul_ps(_mm_mul_ps(u, v), dy),
                            _mm_mul_ps(dx, _mm_add_ps(one, _mm_mul_ps(u, u)))),
                    _mm_sub_ps(_mm_mul_ps(u, dy), _mm_mul_ps(v, dx)),
                    _mm_mul_ps(a, _mm_sub_ps(b0, _mm_load_ps(buf_warped_refColor + i))),
                    minusOne,
                    _mm_load_ps(buf_warped_residual + i),
                    _mm_load_ps(buf_warped_weight + i));
        }

        acc.finish();
        H_out = acc.H.topLeftCorner<8, 8>().cast<double>() * (1.0f / n);
        b_out = acc.H.topRightCorner<8, 1>().cast<double>() * (1.0f / n);

        H_out.block<8, 3>(0, 0) *= SCALE_XI_ROT;
        H_out.block<8, 3>(0, 3) *= SCALE_XI_TRANS;
        H_out.block<8, 1>(0, 6) *= SCALE_A;
        H_out.block<8, 1>(0, 7) *= SCALE_B;
        H_out.block<3, 8>(0, 0) *= SCALE_XI_ROT;
        H_out.block<3, 8>(3, 0) *= SCALE_XI_TRANS;
        H_out.block<1, 8>(6, 0) *= SCALE_A;
        H_out.block<1, 8>(7, 0) *= SCALE_B;
        b_out.segment<3>(0) *= SCALE_XI_ROT;
        b_out.segment<3>(3) *= SCALE_XI_TRANS;
        b_out.segment<1>(6) *= SCALE_A;
        b_out.segment<1>(7) *= SCALE_B;
    }

    // ============================================================================== //
    // Coarse distance map

    CoarseDistanceMap::CoarseDistanceMap(int ww, int hh) {

        fwdWarpedIDDistFinal = new float[ww * hh / 4];
        bfsList1 = new Eigen::Vector2i[ww * hh / 4];
        bfsList2 = new Eigen::Vector2i[ww * hh / 4];
        int fac = 1 << (pyrLevelsUsed - 1);
        coarseProjectionGrid = new PointFrameResidual *[2048 * (ww * hh / (fac * fac))];
        coarseProjectionGridNum = new int[ww * hh / (fac * fac)];
        w[0] = h[0] = 0;

    }

    CoarseDistanceMap::~CoarseDistanceMap() {
        delete[] fwdWarpedIDDistFinal;
        delete[] bfsList1;
        delete[] bfsList2;
        delete[] coarseProjectionGrid;
        delete[] coarseProjectionGridNum;
    }

    void CoarseDistanceMap::makeK(shared_ptr<CalibHessian> HCalib) {

        w[0] = wG[0];
        h[0] = hG[0];

        fx[0] = HCalib->fxl();
        fy[0] = HCalib->fyl();
        cx[0] = HCalib->cxl();
        cy[0] = HCalib->cyl();

        for (int level = 1; level < pyrLevelsUsed; ++level) {
            w[level] = w[0] >> level;
            h[level] = h[0] >> level;
            fx[level] = fx[level - 1] * 0.5;
            fy[level] = fy[level - 1] * 0.5;
            cx[level] = (cx[0] + 0.5) / ((int) 1 << level) - 0.5;
            cy[level] = (cy[0] + 0.5) / ((int) 1 << level) - 0.5;
        }

        for (int level = 0; level < pyrLevelsUsed; ++level) {
            K[level] << fx[level], 0.0, cx[level], 0.0, fy[level], cy[level], 0.0, 0.0, 1.0;
            Ki[level] = K[level].inverse();
            fxi[level] = Ki[level](0, 0);
            fyi[level] = Ki[level](1, 1);
            cxi[level] = Ki[level](0, 2);
            cyi[level] = Ki[level](1, 2);
        }
    }

    void CoarseDistanceMap::makeDistanceMap(std::vector<shared_ptr<FrameHessian>> &frameHessians,
                                            shared_ptr<FrameHessian> frame) {

        int w1 = w[1];
        int h1 = h[1];
        int wh1 = w1 * h1;
        for (int i = 0; i < wh1; i++)
            fwdWarpedIDDistFinal[i] = 1000;


        // make coarse tracking templates for latstRef.
        int numItems = 0;

        for (auto fh : frameHessians) {
            if (frame == fh) continue;

            SE3 fhToNew = frame->PRE_worldToCam * fh->PRE_camToWorld;
            Mat33f KRKi = (K[1] * fhToNew.rotationMatrix().cast<float>() * Ki[0]);
            Vec3f Kt = (K[1] * fhToNew.translation().cast<float>());

            for (auto feat: fh->frame->features) {
                if (feat->point && feat->point->status == Point::PointStatus::ACTIVE) {
                    auto ph = feat->point->mpPH;
                    Vec3f ptp = KRKi * Vec3f(ph->u, ph->v, 1) + Kt * ph->idepth_scaled;
                    int u = ptp[0] / ptp[2] + 0.5f;
                    int v = ptp[1] / ptp[2] + 0.5f;
                    if (!(u > 0 && v > 0 && u < w[1] && v < h[1])) continue;
                    fwdWarpedIDDistFinal[u + w1 * v] = 0;
                    bfsList1[numItems] = Eigen::Vector2i(u, v);
                    numItems++;
                }
            }
        }

        growDistBFS(numItems);
    }

    void CoarseDistanceMap::growDistBFS(int bfsNum) {

        assert(w[0] != 0);
        int w1 = w[1], h1 = h[1];
        for (int k = 1; k < 40; k++) {
            int bfsNum2 = bfsNum;
            std::swap<Eigen::Vector2i *>(bfsList1, bfsList2);
            bfsNum = 0;

            if (k % 2 == 0) {
                for (int i = 0; i < bfsNum2; i++) {
                    int x = bfsList2[i][0];
                    int y = bfsList2[i][1];
                    if (x == 0 || y == 0 || x == w1 - 1 || y == h1 - 1) continue;
                    int idx = x + y * w1;

                    if (fwdWarpedIDDistFinal[idx + 1] > k) {
                        fwdWarpedIDDistFinal[idx + 1] = k;
                        bfsList1[bfsNum] = Eigen::Vector2i(x + 1, y);
                        bfsNum++;
                    }
                    if (fwdWarpedIDDistFinal[idx - 1] > k) {
                        fwdWarpedIDDistFinal[idx - 1] = k;
                        bfsList1[bfsNum] = Eigen::Vector2i(x - 1, y);
                        bfsNum++;
                    }
                    if (fwdWarpedIDDistFinal[idx + w1] > k) {
                        fwdWarpedIDDistFinal[idx + w1] = k;
                        bfsList1[bfsNum] = Eigen::Vector2i(x, y + 1);
                        bfsNum++;
                    }
                    if (fwdWarpedIDDistFinal[idx - w1] > k) {
                        fwdWarpedIDDistFinal[idx - w1] = k;
                        bfsList1[bfsNum] = Eigen::Vector2i(x, y - 1);
                        bfsNum++;
                    }
                }
            } else {
                for (int i = 0; i < bfsNum2; i++) {
                    int x = bfsList2[i][0];
                    int y = bfsList2[i][1];
                    if (x == 0 || y == 0 || x == w1 - 1 || y == h1 - 1) continue;
                    int idx = x + y * w1;

                    if (fwdWarpedIDDistFinal[idx + 1] > k) {
                        fwdWarpedIDDistFinal[idx + 1] = k;
                        bfsList1[bfsNum] = Eigen::Vector2i(x + 1, y);
                        bfsNum++;
                    }
                    if (fwdWarpedIDDistFinal[idx - 1] > k) {
                        fwdWarpedIDDistFinal[idx - 1] = k;
                        bfsList1[bfsNum] = Eigen::Vector2i(x - 1, y);
                        bfsNum++;
                    }
                    if (fwdWarpedIDDistFinal[idx + w1] > k) {
                        fwdWarpedIDDistFinal[idx + w1] = k;
                        bfsList1[bfsNum] = Eigen::Vector2i(x, y + 1);
                        bfsNum++;
                    }
                    if (fwdWarpedIDDistFinal[idx - w1] > k) {
                        fwdWarpedIDDistFinal[idx - w1] = k;
                        bfsList1[bfsNum] = Eigen::Vector2i(x, y - 1);
                        bfsNum++;
                    }

                    if (fwdWarpedIDDistFinal[idx + 1 + w1] > k) {
                        fwdWarpedIDDistFinal[idx + 1 + w1] = k;
                        bfsList1[bfsNum] = Eigen::Vector2i(x + 1, y + 1);
                        bfsNum++;
                    }
                    if (fwdWarpedIDDistFinal[idx - 1 + w1] > k) {
                        fwdWarpedIDDistFinal[idx - 1 + w1] = k;
                        bfsList1[bfsNum] = Eigen::Vector2i(x - 1, y + 1);
                        bfsNum++;
                    }
                    if (fwdWarpedIDDistFinal[idx - 1 - w1] > k) {
                        fwdWarpedIDDistFinal[idx - 1 - w1] = k;
                        bfsList1[bfsNum] = Eigen::Vector2i(x - 1, y - 1);
                        bfsNum++;
                    }
                    if (fwdWarpedIDDistFinal[idx + 1 - w1] > k) {
                        fwdWarpedIDDistFinal[idx + 1 - w1] = k;
                        bfsList1[bfsNum] = Eigen::Vector2i(x + 1, y - 1);
                        bfsNum++;
                    }
                }
            }
        }
    }

    void CoarseDistanceMap::addIntoDistFinal(int u, int v) {
        if (w[0] == 0) return;
        bfsList1[0] = Eigen::Vector2i(u, v);
        fwdWarpedIDDistFinal[u + w[1] * v] = 0;
        growDistBFS(1);
    }

}
