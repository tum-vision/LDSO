#include "Feature.h"
#include "internal/OptimizationBackend/EnergyFunctional.h"
#include "internal/GlobalFuncs.h"

namespace ldso {

    namespace internal {

        bool EFAdjointsValid = false;
        bool EFIndicesValid = false;
        bool EFDeltaValid = false;

        EnergyFunctional::EnergyFunctional() :
                accSSE_top_L(new AccumulatedTopHessianSSE),
                accSSE_top_A(new AccumulatedTopHessianSSE),
                accSSE_bot(new AccumulatedSCHessianSSE) {}

        EnergyFunctional::~EnergyFunctional() {
            if (adHost != 0) delete[] adHost;
            if (adTarget != 0) delete[] adTarget;
            if (adHostF != 0) delete[] adHostF;
            if (adTargetF != 0) delete[] adTargetF;
            if (adHTdeltaF != 0) delete[] adHTdeltaF;
        }

        void EnergyFunctional::insertResidual(shared_ptr<PointFrameResidual> r) {
            r->takeData();
            connectivityMap[(((uint64_t) r->host.lock()->frameID) << 32) + ((uint64_t) r->target.lock()->frameID)][0]++;
            nResiduals++;
        }

        void EnergyFunctional::insertFrame(shared_ptr<FrameHessian> fh, shared_ptr<CalibHessian> Hcalib) {
            fh->takeData();
            frames.push_back(fh);
            fh->idx = frames.size();
            nFrames++;

            // extend H,b
            assert(HM.cols() == 8 * nFrames + CPARS - 8);
            bM.conservativeResize(8 * nFrames + CPARS);
            HM.conservativeResize(8 * nFrames + CPARS, 8 * nFrames + CPARS);
            bM.tail<8>().setZero();
            HM.rightCols<8>().setZero();
            HM.bottomRows<8>().setZero();

            // set index as invalid
            EFIndicesValid = false;
            EFAdjointsValid = false;
            EFDeltaValid = false;

            setAdjointsF(Hcalib);
            makeIDX();

            // set connectivity map
            for (auto fh2: frames) {
                connectivityMap[(((uint64_t) fh->frameID) << 32) + ((uint64_t) fh2->frameID)] = Eigen::Vector2i(0, 0);
                if (fh2 != fh)
                    connectivityMap[(((uint64_t) fh2->frameID) << 32) + ((uint64_t) fh->frameID)] = Eigen::Vector2i(0,
                                                                                                                    0);
            }
        }

        void EnergyFunctional::dropResidual(shared_ptr<PointFrameResidual> r) {

            // remove this residual from pointHessian->residualsAll
            shared_ptr<PointHessian> p = r->point.lock();
            deleteOut<PointFrameResidual>(p->residuals, r);
            connectivityMap[(((uint64_t) r->host.lock()->frameID) << 32) + ((uint64_t) r->target.lock()->frameID)][0]--;
            nResiduals--;
        }

        void EnergyFunctional::marginalizeFrame(shared_ptr<FrameHessian> fh) {

            assert(EFDeltaValid);
            assert(EFAdjointsValid);
            assert(EFIndicesValid);

            int ndim = nFrames * 8 + CPARS - 8;// new dimension
            int odim = nFrames * 8 + CPARS;// old dimension

            if ((int) fh->idx != (int) frames.size() - 1) {
                int io = fh->idx * 8 + CPARS;    // index of frame to move to end
                int ntail = 8 * (nFrames - fh->idx - 1);
                assert((io + 8 + ntail) == nFrames * 8 + CPARS);

                Vec8 bTmp = bM.segment<8>(io);
                VecX tailTMP = bM.tail(ntail);
                bM.segment(io, ntail) = tailTMP;
                bM.tail<8>() = bTmp;

                MatXX HtmpCol = HM.block(0, io, odim, 8);
                MatXX rightColsTmp = HM.rightCols(ntail);
                HM.block(0, io, odim, ntail) = rightColsTmp;
                HM.rightCols(8) = HtmpCol;

                MatXX HtmpRow = HM.block(io, 0, 8, odim);
                MatXX botRowsTmp = HM.bottomRows(ntail);
                HM.block(io, 0, ntail, odim) = botRowsTmp;
                HM.bottomRows(8) = HtmpRow;
            }


            // marginalize. First add prior here, instead of to active.
            HM.bottomRightCorner<8, 8>().diagonal() += fh->prior;
            bM.tail<8>() += fh->prior.cwiseProduct(fh->delta_prior);

            VecX SVec = (HM.diagonal().cwiseAbs() + VecX::Constant(HM.cols(), 10)).cwiseSqrt();
            VecX SVecI = SVec.cwiseInverse();

            // scale!
            MatXX HMScaled = SVecI.asDiagonal() * HM * SVecI.asDiagonal();
            VecX bMScaled = SVecI.asDiagonal() * bM;

            // invert bottom part!
            Mat88 hpi = HMScaled.bottomRightCorner<8, 8>();
            hpi = 0.5f * (hpi + hpi);
            hpi = hpi.inverse();
            hpi = 0.5f * (hpi + hpi);

            // schur-complement!
            MatXX bli = HMScaled.bottomLeftCorner(8, ndim).transpose() * hpi;
            HMScaled.topLeftCorner(ndim, ndim).noalias() -= bli * HMScaled.bottomLeftCorner(8, ndim);
            bMScaled.head(ndim).noalias() -= bli * bMScaled.tail<8>();

            // unscale!
            HMScaled = SVec.asDiagonal() * HMScaled * SVec.asDiagonal();
            bMScaled = SVec.asDiagonal() * bMScaled;

            // set.
            HM = 0.5 * (HMScaled.topLeftCorner(ndim, ndim) + HMScaled.topLeftCorner(ndim, ndim).transpose());
            bM = bMScaled.head(ndim);

            // remove from vector, without changing the order!
            for (unsigned int i = fh->idx; i + 1 < frames.size(); i++) {
                frames[i] = frames[i + 1];
                frames[i]->idx = i;
            }
            frames.pop_back();
            nFrames--;

            assert((int) frames.size() * 8 + CPARS == (int) HM.rows());
            assert((int) frames.size() * 8 + CPARS == (int) HM.cols());
            assert((int) frames.size() * 8 + CPARS == (int) bM.size());
            assert((int) frames.size() == (int) nFrames);

            EFIndicesValid = false;
            EFAdjointsValid = false;
            EFDeltaValid = false;

            makeIDX();
        }

        void EnergyFunctional::removePoint(shared_ptr<PointHessian> ph) {
            for (auto &r: ph->residuals) {
                connectivityMap[(((uint64_t) r->host.lock()->frameID) << 32) +
                                ((uint64_t) r->target.lock()->frameID)][0]--;
                nResiduals--;
            }
            ph->residuals.clear();
            if (!ph->alreadyRemoved)
                nPoints--;
            EFIndicesValid = false;
        }

        void EnergyFunctional::marginalizePointsF() {

            allPointsToMarg.clear();

            // go through all points to see which to marg
            for (auto f: frames) {
                for (shared_ptr<Feature> feat: f->frame->features) {

                    if (feat->status == Feature::FeatureStatus::VALID &&
                        feat->point->status == Point::PointStatus::MARGINALIZED) {
                        shared_ptr<PointHessian> p = feat->point->mpPH;
                        p->priorF *= setting_idepthFixPriorMargFac;
                        for (auto r: p->residuals)
                            if (r->isActive())
                                connectivityMap[(((uint64_t) r->host.lock()->frameID) << 32) +
                                                ((uint64_t) r->target.lock()->frameID)][1]++;
                        allPointsToMarg.push_back(p);
                    }
                }
            }

            accSSE_bot->setZero(nFrames);
            accSSE_top_A->setZero(nFrames);

            for (auto p : allPointsToMarg) {
                accSSE_top_A->addPoint<2>(p, this);
                accSSE_bot->addPoint(p, false);
                removePoint(p);
            }

            MatXX M, Msc;
            VecX Mb, Mbsc;
            accSSE_top_A->stitchDouble(M, Mb, this, false, false);
            accSSE_bot->stitchDouble(Msc, Mbsc, this);

            resInM += accSSE_top_A->nres[0];

            MatXX H = M - Msc;
            VecX b = Mb - Mbsc;

            if (setting_solverMode & SOLVER_ORTHOGONALIZE_POINTMARG) {
                bool haveFirstFrame = false;
                for (auto f:frames)
                    if (f->frameID == 0)
                        haveFirstFrame = true;
                if (!haveFirstFrame)
                    orthogonalize(&bM, &HM);
            }

            HM += setting_margWeightFac * H;
            bM += setting_margWeightFac * b;

            if (setting_solverMode & SOLVER_ORTHOGONALIZE_FULL)
                orthogonalize(&bM, &HM);

            EFIndicesValid = false;
            makeIDX();
        }

        void EnergyFunctional::dropPointsF() {

            for (auto f: frames) {
                for (shared_ptr<Feature> feat: f->frame->features) {
                    if (feat->point &&
                        (feat->point->status == Point::PointStatus::OUTLIER ||
                         feat->point->status == Point::PointStatus::OUT)
                        && feat->point->mpPH->alreadyRemoved == false) {
                        removePoint(feat->point->mpPH);
                    }
                }
            }
            EFIndicesValid = false;
            makeIDX();
        }

        void EnergyFunctional::solveSystemF(int iteration, double lambda, shared_ptr<CalibHessian> HCalib) {

            if (setting_solverMode & SOLVER_USE_GN) lambda = 0;
            if (setting_solverMode & SOLVER_FIX_LAMBDA) lambda = 1e-5;

            assert(EFDeltaValid);
            assert(EFAdjointsValid);
            assert(EFIndicesValid);

            // construct matricies
            MatXX HL_top, HA_top, H_sc;
            VecX bL_top, bA_top, bM_top, b_sc;

            accumulateAF_MT(HA_top, bA_top, multiThreading);
            accumulateLF_MT(HL_top, bL_top, multiThreading);
            accumulateSCF_MT(H_sc, b_sc, multiThreading);

            bM_top = (bM + HM * getStitchedDeltaF());

            MatXX HFinal_top;
            VecX bFinal_top;

            if (setting_solverMode & SOLVER_ORTHOGONALIZE_SYSTEM) {
                // have a look if prior is there.
                bool haveFirstFrame = false;
                for (auto f : frames)
                    if (f->frameID == 0)
                        haveFirstFrame = true;

                MatXX HT_act = HL_top + HA_top - H_sc;
                VecX bT_act = bL_top + bA_top - b_sc;

                if (!haveFirstFrame)
                    orthogonalize(&bT_act, &HT_act);

                HFinal_top = HT_act + HM;
                bFinal_top = bT_act + bM_top;
                lastHS = HFinal_top;
                lastbS = bFinal_top;

                for (int i = 0; i < 8 * nFrames + CPARS; i++)
                    HFinal_top(i, i) *= (1 + lambda);
            } else {
                HFinal_top = HL_top + HM + HA_top;
                bFinal_top = bL_top + bM_top + bA_top - b_sc;

                lastHS = HFinal_top - H_sc;
                lastbS = bFinal_top;

                for (int i = 0; i < 8 * nFrames + CPARS; i++)
                    HFinal_top(i, i) *= (1 + lambda);
                HFinal_top -= H_sc * (1.0f / (1 + lambda));
            }

            // get the result
            VecX x;
            if (setting_solverMode & SOLVER_SVD) {
                VecX SVecI = HFinal_top.diagonal().cwiseSqrt().cwiseInverse();
                MatXX HFinalScaled = SVecI.asDiagonal() * HFinal_top * SVecI.asDiagonal();
                VecX bFinalScaled = SVecI.asDiagonal() * bFinal_top;
                Eigen::JacobiSVD<MatXX> svd(HFinalScaled, Eigen::ComputeThinU | Eigen::ComputeThinV);

                VecX S = svd.singularValues();
                double minSv = 1e10, maxSv = 0;
                for (int i = 0; i < S.size(); i++) {
                    if (S[i] < minSv) minSv = S[i];
                    if (S[i] > maxSv) maxSv = S[i];
                }

                VecX Ub = svd.matrixU().transpose() * bFinalScaled;
                int setZero = 0;
                for (int i = 0; i < Ub.size(); i++) {
                    if (S[i] < setting_solverModeDelta * maxSv) {
                        Ub[i] = 0;
                        setZero++;
                    }

                    if ((setting_solverMode & SOLVER_SVD_CUT7) && (i >= Ub.size() - 7)) {
                        Ub[i] = 0;
                        setZero++;
                    } else Ub[i] /= S[i];
                }
                x = SVecI.asDiagonal() * svd.matrixV() * Ub;

            } else {

                VecX SVecI = (HFinal_top.diagonal() + VecX::Constant(HFinal_top.cols(), 10)).cwiseSqrt().cwiseInverse();
                MatXX HFinalScaled = SVecI.asDiagonal() * HFinal_top * SVecI.asDiagonal();

                /*
                x = SVecI.asDiagonal() * HFinalScaled.ldlt().solve(
                        SVecI.asDiagonal() * bFinal_top);//  SVec.asDiagonal() * svd.matrixV() * Ub;
                        */

                x = SVecI.asDiagonal() * HFinalScaled.ldlt().solve(
                        SVecI.asDiagonal() * bFinal_top);//  SVec.asDiagonal() * svd.matrixV() * Ub;

            }

            if ((setting_solverMode & SOLVER_ORTHOGONALIZE_X) ||
                (iteration >= 2 && (setting_solverMode & SOLVER_ORTHOGONALIZE_X_LATER))) {
                VecX xOld = x;
                orthogonalize(&x, 0);
            }

            lastX = x;

            currentLambda = lambda;
            resubstituteF_MT(x, HCalib, multiThreading);
            currentLambda = 0;

        }

        double EnergyFunctional::calcMEnergyF() {
            assert(EFDeltaValid);
            assert(EFAdjointsValid);
            assert(EFIndicesValid);
            VecX delta = getStitchedDeltaF();
            return delta.dot(2 * bM + HM * delta);
        }

        double EnergyFunctional::calcLEnergyF_MT() {
            assert(EFDeltaValid);
            assert(EFAdjointsValid);
            assert(EFIndicesValid);

            double E = 0;
            for (auto f : frames)
                E += f->delta_prior.cwiseProduct(f->prior).dot(f->delta_prior);

            E += cDeltaF.cwiseProduct(cPriorF).dot(cDeltaF);

            red->reduce(bind(&EnergyFunctional::calcLEnergyPt,
                             this, _1, _2, _3, _4), 0, allPoints.size(), 50);

            // E += calcLEnergyFeat(); // calc feature's energy

            return E + red->stats[0];
        }

        void EnergyFunctional::makeIDX() {

            for (unsigned int idx = 0; idx < frames.size(); idx++)
                frames[idx]->idx = idx;

            allPoints.clear();

            for (auto f: frames) {
                for (shared_ptr<Feature> feat: f->frame->features) {
                    if (feat->status == Feature::FeatureStatus::VALID &&
                        feat->point->status == Point::PointStatus::ACTIVE) {
                        shared_ptr<PointHessian> p = feat->point->mpPH;
                        allPoints.push_back(p);
                        for (auto &r : p->residuals) {
                            r->hostIDX = r->host.lock()->idx;
                            r->targetIDX = r->target.lock()->idx;
                        }
                    }
                }
            }
            EFIndicesValid = true;
        }

        void EnergyFunctional::setDeltaF(shared_ptr<CalibHessian> HCalib) {
            if (adHTdeltaF != 0) delete[] adHTdeltaF;
            adHTdeltaF = new Mat18f[nFrames * nFrames];
            for (int h = 0; h < nFrames; h++)
                for (int t = 0; t < nFrames; t++) {
                    int idx = h + t * nFrames;
                    adHTdeltaF[idx] =
                            frames[h]->get_state_minus_stateZero().head<8>().cast<float>().transpose() * adHostF[idx]
                            +
                            frames[t]->get_state_minus_stateZero().head<8>().cast<float>().transpose() * adTargetF[idx];
                }

            cDeltaF = HCalib->value_minus_value_zero.cast<float>();
            for (auto f : frames) {
                f->delta = f->get_state_minus_stateZero().head<8>();
                f->delta_prior = (f->get_state() - f->getPriorZero()).head<8>();

                for (auto feat: f->frame->features) {
                    if (feat->status == Feature::FeatureStatus::VALID && feat->point &&
                        feat->point->status == Point::PointStatus::ACTIVE) {
                        auto p = feat->point->mpPH;
                        p->deltaF = p->idepth - p->idepth_zero;
                    }
                }
            }
            EFDeltaValid = true;
        }

        void EnergyFunctional::setAdjointsF(shared_ptr<CalibHessian> Hcalib) {

            if (adHost != 0) delete[] adHost;
            if (adTarget != 0) delete[] adTarget;

            adHost = new Mat88[nFrames * nFrames];
            adTarget = new Mat88[nFrames * nFrames];

            for (int h = 0; h < nFrames; h++)
                for (int t = 0; t < nFrames; t++) {
                    shared_ptr<FrameHessian> host = frames[h];
                    shared_ptr<FrameHessian> target = frames[t];

                    SE3 hostToTarget = target->get_worldToCam_evalPT() * host->get_worldToCam_evalPT().inverse();

                    Mat88 AH = Mat88::Identity();
                    Mat88 AT = Mat88::Identity();

                    AH.topLeftCorner<6, 6>() = -hostToTarget.Adj().transpose();
                    AT.topLeftCorner<6, 6>() = Mat66::Identity();


                    Vec2f affLL = AffLight::fromToVecExposure(host->ab_exposure, target->ab_exposure, host->aff_g2l_0(),
                                                              target->aff_g2l_0()).cast<float>();
                    AT(6, 6) = -affLL[0];
                    AH(6, 6) = affLL[0];
                    AT(7, 7) = -1;
                    AH(7, 7) = affLL[0];

                    AH.block<3, 8>(0, 0) *= SCALE_XI_TRANS;
                    AH.block<3, 8>(3, 0) *= SCALE_XI_ROT;
                    AH.block<1, 8>(6, 0) *= SCALE_A;
                    AH.block<1, 8>(7, 0) *= SCALE_B;
                    AT.block<3, 8>(0, 0) *= SCALE_XI_TRANS;
                    AT.block<3, 8>(3, 0) *= SCALE_XI_ROT;
                    AT.block<1, 8>(6, 0) *= SCALE_A;
                    AT.block<1, 8>(7, 0) *= SCALE_B;

                    adHost[h + t * nFrames] = AH;
                    adTarget[h + t * nFrames] = AT;
                }

            cPrior = VecC::Constant(setting_initialCalibHessian);


            if (adHostF != 0) delete[] adHostF;
            if (adTargetF != 0) delete[] adTargetF;
            adHostF = new Mat88f[nFrames * nFrames];
            adTargetF = new Mat88f[nFrames * nFrames];

            for (int h = 0; h < nFrames; h++)
                for (int t = 0; t < nFrames; t++) {
                    adHostF[h + t * nFrames] = adHost[h + t * nFrames].cast<float>();
                    adTargetF[h + t * nFrames] = adTarget[h + t * nFrames].cast<float>();
                }

            cPriorF = cPrior.cast<float>();
            EFAdjointsValid = true;
        }

        void EnergyFunctional::resubstituteF_MT(const VecX &x, shared_ptr<CalibHessian> HCalib, bool MT) {
            assert(x.size() == CPARS + nFrames * 8);

            VecXf xF = x.cast<float>();
            HCalib->step = -x.head<CPARS>();

            Mat18f *xAd = new Mat18f[nFrames * nFrames];
            VecCf cstep = xF.head<CPARS>();
            for (auto h : frames) {
                h->step.head<8>() = -x.segment<8>(CPARS + 8 * h->idx);
                h->step.tail<2>().setZero();

                for (auto t : frames)
                    xAd[nFrames * h->idx + t->idx] =
                            xF.segment<8>(CPARS + 8 * h->idx).transpose() * adHostF[h->idx + nFrames * t->idx]
                            + xF.segment<8>(CPARS + 8 * t->idx).transpose() * adTargetF[h->idx + nFrames * t->idx];
            }

            if (MT)
                red->reduce(bind(&EnergyFunctional::resubstituteFPt,
                                 this, cstep, xAd, _1, _2, _3, _4), 0, allPoints.size(), 50);
            else
                resubstituteFPt(cstep, xAd, 0, allPoints.size(), 0, 0);

            delete[] xAd;
        }

        void EnergyFunctional::resubstituteFPt(const VecCf &xc, Mat18f *xAd, int min, int max, Vec10 *stats, int tid) {

            for (int k = min; k < max; k++) {
                auto p = allPoints[k];

                int ngoodres = 0;
                for (auto r : p->residuals)
                    if (r->isActive())
                        ngoodres++;

                if (ngoodres == 0) {
                    p->step = 0;
                    continue;
                }

                float b = p->bdSumF;
                b -= xc.dot(p->Hcd_accAF + p->Hcd_accLF);

                for (auto r : p->residuals) {
                    if (!r->isActive()) continue;
                    b -= xAd[r->hostIDX * nFrames + r->targetIDX] * r->JpJdF;
                }

                if (!std::isfinite(b) || std::isnan(b)) {
                    return;
                }

                p->step = -b * p->HdiF;
            }
        }

        // accumulates & shifts L.
        void EnergyFunctional::accumulateAF_MT(MatXX &H, VecX &b, bool MT) {
            if (MT) {
                red->reduce(bind(&AccumulatedTopHessianSSE::setZero, accSSE_top_A, nFrames, _1, _2, _3, _4), 0,
                            0, 0);
                red->reduce(bind(&AccumulatedTopHessianSSE::addPointsInternal<0>,
                                 accSSE_top_A, &allPoints, this, _1, _2, _3, _4), 0, allPoints.size(), 50);
                accSSE_top_A->stitchDoubleMT(red, H, b, this, false, true);
                resInA = accSSE_top_A->nres[0];
            } else {
                accSSE_top_A->setZero(nFrames);
                int cntPointAdded = 0;
                for (auto f : frames) {
                    for (shared_ptr<Feature> &feat: f->frame->features) {
                        if (feat->status == Feature::FeatureStatus::VALID && feat->point &&
                            feat->point->status == Point::PointStatus::ACTIVE) {
                            auto p = feat->point->mpPH;
                            accSSE_top_A->addPoint<0>(p, this);
                            cntPointAdded++;
                        }
                    }
                }
                accSSE_top_A->stitchDoubleMT(red, H, b, this, false, false);
                resInA = accSSE_top_A->nres[0];
            }
        }

        // accumulates & shifts L.
        void EnergyFunctional::accumulateLF_MT(MatXX &H, VecX &b, bool MT) {
            if (MT) {
                red->reduce(bind(&AccumulatedTopHessianSSE::setZero, accSSE_top_L, nFrames, _1, _2, _3, _4), 0,
                            0, 0);
                red->reduce(bind(&AccumulatedTopHessianSSE::addPointsInternal<1>,
                                 accSSE_top_L, &allPoints, this, _1, _2, _3, _4), 0, allPoints.size(), 50);
                accSSE_top_L->stitchDoubleMT(red, H, b, this, true, true);
                resInL = accSSE_top_L->nres[0];
            } else {
                accSSE_top_L->setZero(nFrames);
                int cntPointAdded = 0;
                for (auto f : frames) {
                    for (auto feat: f->frame->features) {
                        if (feat->status == Feature::FeatureStatus::VALID &&
                            feat->point->status == Point::PointStatus::ACTIVE) {
                            auto p = feat->point->mpPH;
                            accSSE_top_L->addPoint<1>(p, this);
                            cntPointAdded++;
                        }
                    }
                }
                accSSE_top_L->stitchDoubleMT(red, H, b, this, true, false);
                resInL = accSSE_top_L->nres[0];
            }
        }

        void EnergyFunctional::accumulateSCF_MT(MatXX &H, VecX &b, bool MT) {
            if (MT) {
                red->reduce(bind(&AccumulatedSCHessianSSE::setZero, accSSE_bot, nFrames, _1, _2, _3, _4), 0, 0,
                            0);
                red->reduce(bind(&AccumulatedSCHessianSSE::addPointsInternal,
                                 accSSE_bot, &allPoints, true, _1, _2, _3, _4), 0, allPoints.size(), 50);
                accSSE_bot->stitchDoubleMT(red, H, b, this, true);
            } else {
                accSSE_bot->setZero(nFrames);
                int cntPointAdded = 0;
                for (auto f : frames) {
                    for (auto feat: f->frame->features) {
                        if (feat->status == Feature::FeatureStatus::VALID &&
                            feat->point->status == Point::PointStatus::ACTIVE) {
                            auto p = feat->point->mpPH;
                            accSSE_bot->addPoint(p, true);
                            cntPointAdded++;
                        }
                    }
                }
                accSSE_bot->stitchDoubleMT(red, H, b, this, false);
            }
        }

        void EnergyFunctional::calcLEnergyPt(int min, int max, Vec10 *stats, int tid) {

            Accumulator11 E;
            E.initialize();
            VecCf dc = cDeltaF;

            for (int i = min; i < max; i++) {
                auto p = allPoints[i];
                float dd = p->deltaF;

                for (auto r : p->residuals) {
                    if (!r->isLinearized || !r->isActive()) continue;

                    Mat18f dp = adHTdeltaF[r->hostIDX + nFrames * r->targetIDX];
                    shared_ptr<RawResidualJacobian> rJ = r->J;

                    // compute Jp*delta
                    float Jp_delta_x_1 = rJ->Jpdxi[0].dot(dp.head<6>())
                                         + rJ->Jpdc[0].dot(dc)
                                         + rJ->Jpdd[0] * dd;

                    float Jp_delta_y_1 = rJ->Jpdxi[1].dot(dp.head<6>())
                                         + rJ->Jpdc[1].dot(dc)
                                         + rJ->Jpdd[1] * dd;

                    __m128 Jp_delta_x = _mm_set1_ps(Jp_delta_x_1);
                    __m128 Jp_delta_y = _mm_set1_ps(Jp_delta_y_1);
                    __m128 delta_a = _mm_set1_ps((float) (dp[6]));
                    __m128 delta_b = _mm_set1_ps((float) (dp[7]));

                    for (int i = 0; i + 3 < patternNum; i += 4) {
                        // PATTERN: E = (2*res_toZeroF + J*delta) * J*delta.
                        __m128 Jdelta = _mm_mul_ps(_mm_load_ps(((float *) (rJ->JIdx)) + i), Jp_delta_x);
                        Jdelta = _mm_add_ps(Jdelta,
                                            _mm_mul_ps(_mm_load_ps(((float *) (rJ->JIdx + 1)) + i), Jp_delta_y));
                        Jdelta = _mm_add_ps(Jdelta, _mm_mul_ps(_mm_load_ps(((float *) (rJ->JabF)) + i), delta_a));
                        Jdelta = _mm_add_ps(Jdelta, _mm_mul_ps(_mm_load_ps(((float *) (rJ->JabF + 1)) + i), delta_b));

                        __m128 r0 = _mm_load_ps(((float *) &r->res_toZeroF) + i);
                        r0 = _mm_add_ps(r0, r0);
                        r0 = _mm_add_ps(r0, Jdelta);
                        Jdelta = _mm_mul_ps(Jdelta, r0);
                        E.updateSSENoShift(Jdelta);
                    }
                    for (int i = ((patternNum >> 2) << 2); i < patternNum; i++) {
                        float Jdelta = rJ->JIdx[0][i] * Jp_delta_x_1 + rJ->JIdx[1][i] * Jp_delta_y_1 +
                                       rJ->JabF[0][i] * dp[6] + rJ->JabF[1][i] * dp[7];
                        E.updateSingleNoShift((float) (Jdelta * (Jdelta + 2 * r->res_toZeroF[i])));
                    }
                }

                E.updateSingle(p->deltaF * p->deltaF * p->priorF);
            }
            E.finish();
            (*stats)[0] += E.A;
        }


        void EnergyFunctional::orthogonalize(VecX *b, MatXX *H) {

            std::vector<VecX> ns;
            ns.insert(ns.end(), lastNullspaces_pose.begin(), lastNullspaces_pose.end());
            ns.insert(ns.end(), lastNullspaces_scale.begin(), lastNullspaces_scale.end());

            // make Nullspaces matrix
            MatXX N(ns[0].rows(), ns.size());
            for (unsigned int i = 0; i < ns.size(); i++)
                N.col(i) = ns[i].normalized();

            // compute Npi := N * (N' * N)^-1 = pseudo inverse of N.
            Eigen::JacobiSVD<MatXX> svdNN(N, Eigen::ComputeThinU | Eigen::ComputeThinV);

            VecX SNN = svdNN.singularValues();
            double minSv = 1e10, maxSv = 0;
            for (int i = 0; i < SNN.size(); i++) {
                if (SNN[i] < minSv) minSv = SNN[i];
                if (SNN[i] > maxSv) maxSv = SNN[i];
            }
            for (int i = 0; i < SNN.size(); i++) {
                if (SNN[i] > setting_solverModeDelta * maxSv)
                    SNN[i] = 1.0 / SNN[i];
                else SNN[i] = 0;
            }

            MatXX Npi = svdNN.matrixU() * SNN.asDiagonal() * svdNN.matrixV().transpose();    // [dim] x 9.
            MatXX NNpiT = N * Npi.transpose();    // [dim] x [dim].
            MatXX NNpiTS = 0.5 * (NNpiT + NNpiT.transpose());    // = N * (N' * N)^-1 * N'.

            if (b != 0) *b -= NNpiTS * *b;
            if (H != 0) *H -= NNpiTS * *H * NNpiTS;
        }
    }
}