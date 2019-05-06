#include "Feature.h"
#include "internal/PR.h"
#include "internal/GlobalCalib.h"

#include "frontend/LoopClosing.h"
#include "frontend/FeatureMatcher.h"
#include "frontend/FullSystem.h"

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <boost/format.hpp>

#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/linear_solver_eigen.h>
#include <g2o/core/robust_kernel_impl.h>


namespace ldso {

    // -----------------------------------------------------------
    LoopClosing::LoopClosing(FullSystem *fullsystem) :
            kfDB(new DBoW3::Database(*fullsystem->vocab)), voc(fullsystem->vocab),
            globalMap(fullsystem->globalMap), Hcalib(fullsystem->Hcalib->mpCH),
            coarseDistanceMap(fullsystem->GetDistanceMap()),
            fullSystem(fullsystem) {

        mainLoop = thread(&LoopClosing::Run, this);
        idepthMap = new float[wG[0] * hG[0]];
    }

    void LoopClosing::InsertKeyFrame(shared_ptr<Frame> &frame) {
        unique_lock<mutex> lock(mutexKFQueue);
        KFqueue.push_back(frame);
    }

    void LoopClosing::Run() {
        finished = false;

        while (1) {

            if (needFinish) {
                LOG(INFO) << "find loop closing thread need finish flag!" << endl;
                break;
            }

            {
                // get the oldest one
                unique_lock<mutex> lock(mutexKFQueue);
                if (KFqueue.empty()) {
                    lock.unlock();
                    usleep(5000);
                    continue;
                }
                currentKF = KFqueue.front();
                KFqueue.pop_front();

                if (KFqueue.size() > 20)
                    KFqueue.clear();

                allKF.push_back(currentKF);
            }

            currentKF->ComputeBoW(voc);
            if (DetectLoop(currentKF)) {
                bool mapIdle = globalMap->Idle();
                if (CorrectLoop(Hcalib)) {
                    // start a pose graph optimization
                    if (mapIdle) {
                        LOG(INFO) << "call global pose graph!" << endl;
                        bool ret = globalMap->OptimizeALLKFs();
                        if (ret)
                            needPoseGraph = false;
                    } else {
                        LOG(INFO) << "still need pose graph optimization!" << endl;
                        needPoseGraph = true;
                    }
                }
            }


            if (needPoseGraph && globalMap->Idle()) {
                LOG(INFO) << "run another pose graph!" << endl;
                if (globalMap->OptimizeALLKFs())
                    needPoseGraph = false;
            }

            usleep(5000);
        }

        finished = true;
    }

    bool LoopClosing::DetectLoop(shared_ptr<Frame> &frame) {

        DBoW3::QueryResults results;
        kfDB->query(frame->bowVec, results, 1, maxKFId - kfGap);

        if (results.empty()) {
            DBoW3::EntryId id = kfDB->add(frame->bowVec, frame->featVec);
            maxKFId = id;
            checkedKFs[id] = frame;
            return false;
        }

        DBoW3::Result r = results[0];
        candidateKF = checkedKFs[r.Id];

        auto connected = frame->GetConnectedKeyFrames();
        unsigned long minKFId = 9999999, maxKFId = 0;

        for (auto &kf: connected) {
            if (kf->kfId < minKFId)
                minKFId = kf->kfId;
            if (kf->kfId > maxKFId)
                maxKFId = kf->kfId;
        }

        if (candidateKF->kfId <= maxKFId && candidateKF->kfId >= minKFId) {
            // candidate is in active window
            return false;
        }

        LOG(INFO) << "candidate kf id: " << candidateKF->kfId << ", max id: " << maxKFId << ", min id: " << minKFId
                  << endl;

        if (r.Score < minScoreAccept) {
            DBoW3::EntryId id = kfDB->add(frame->bowVec, frame->featVec);
            maxKFId = id;
            checkedKFs[id] = frame;
            candidateKF = checkedKFs[r.Id];
            LOG(INFO) << "add loop candidate from " << candidateKF->kfId << ", current: " << frame->kfId << ", score: "
                      << r.Score << endl;
            return true;
        }

        // detected a possible loop
        candidateKF = checkedKFs[r.Id];
        LOG(INFO) << "add loop candidate from " << candidateKF->kfId << ", current: " << frame->kfId << ", score: "
                  << r.Score << endl;
        return true;   // don't add into database
    }

    bool LoopClosing::CorrectLoop(shared_ptr<CalibHessian> Hcalib) {

        // We compute first ORB matches for each candidate
        FeatureMatcher matcher(0.75, true);
        bool success = false;
        int nCandidates = 0; //candidates with enough matches

        // intrinsics
        cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
        K.at<float>(0, 0) = Hcalib->fxl();
        K.at<float>(1, 1) = Hcalib->fyl();
        K.at<float>(0, 2) = Hcalib->cxl();
        K.at<float>(1, 2) = Hcalib->cyl();

        shared_ptr<Frame> pKF = candidateKF;
        vector<Match> matches;
        int nmatches = matcher.SearchByBoW(currentKF, pKF, matches);

        if (nmatches < 10) {
            LOG(INFO) << "no enough matches: " << nmatches << endl;
        } else {
            LOG(INFO) << "matches: " << nmatches << endl;

            // now we have a candidate proposed by dbow, let's try opencv's solve pnp ransac to see if there are enough inliers
            vector<cv::Point3f> p3d;
            vector<cv::Point2f> p2d;
            cv::Mat inliers;
            vector<int> matchIdx;

            for (size_t k = 0; k < matches.size(); k++) {
                auto &m = matches[k];
                shared_ptr<Feature> &featKF = pKF->features[m.index2];
                shared_ptr<Feature> &featCurrent = currentKF->features[m.index1];

                if (featKF->status == Feature::FeatureStatus::VALID &&
                    featKF->point->status != Point::PointStatus::OUTLIER) {
                    // there should be a 3d point
                    // pt unused?
                    //shared_ptr<Point> &pt = featKF->point;
                    // compute 3d pos in ref
                    Vec3f pt3 = (1.0 / featKF->invD) * Vec3f(
                            Hcalib->fxli() * (featKF->uv[0] - Hcalib->cxl()),
                            Hcalib->fyli() * (featKF->uv[1] - Hcalib->cyl()),
                            1
                    );
                    cv::Point3f pt3d(pt3[0], pt3[1], pt3[2]);
                    p3d.push_back(pt3d);
                    p2d.push_back(cv::Point2f(featCurrent->uv[0], featCurrent->uv[1]));
                    matchIdx.push_back(k);
                }
            }

            if (p3d.size() < 10) {
                LOG(INFO) << "3d points not enough: " << p3d.size() << endl;
                return false;
            }

            cv::Mat R, t;
#if (defined(CV_VERSION_EPOCH) && CV_VERSION_EPOCH == 2)
            // OpenCV 2 has "minInliers" parameter
            cv::solvePnPRansac(p3d, p2d, K, cv::Mat(), R, t, false, 100, 8.0, 0, inliers);
#else
            // OpenCV 3 and 4 has "confidence" parameter
            cv::solvePnPRansac(p3d, p2d, K, cv::Mat(), R, t, false, 100, 8.0, 0.99, inliers);
#endif
            int cntInliers = 0;

            vector<Match> inlierMatches;
            for (int k = 0; k < inliers.rows; k++) {
                inlierMatches.push_back(matches[matchIdx[inliers.at<int>(k, 0)]]);
                cntInliers++;
            }

            if (cntInliers < 10) {
                LOG(INFO) << "Ransac inlier not enough: " << cntInliers << endl;
                return false;
            }

            LOG(INFO) << "Loop detected from kf " << currentKF->kfId << " to " << pKF->kfId
                      << ", inlier matches: " << cntInliers << endl;

            // and then test with the estimated Tcw
            SE3 TcrEsti(
            SO3::exp(Vec3(R.at<double>(0, 0), R.at<double>(1, 0), R.at<double>(2, 0))),
                    Vec3(t.at<double>(0, 0), t.at<double>(1, 0), t.at<double>(2, 0)));

            Sim3 ScrEsti(TcrEsti.matrix());
            ScrEsti.setScale(1.0);

            Mat77 Hessian;

            if (ComputeOptimizedPose(pKF, ScrEsti, Hcalib, Hessian) == false) {
                return false;
            }

            // setup pose graph
            {
                Sim3 SCurRef = ScrEsti;
                unique_lock<mutex> lock(currentKF->mutexPoseRel);
                currentKF->poseRel[pKF] = Frame::RELPOSE(SCurRef, Hessian, true);   // and an pose graph edge
                pKF->poseRel[currentKF] = Frame::RELPOSE(SCurRef.inverse(), Hessian, true);
            }

            success = true;

            if (setting_showLoopClosing && success) {
                LOG(INFO) << "please see loop closing between " << currentKF->kfId << " and " << pKF->kfId << endl;
                setting_pause = true;
                matcher.DrawMatches(currentKF, pKF, inlierMatches);
                setting_pause = false;
            }

            setting_pause = false;
        }
        nCandidates++;
        return success;
    }

    bool LoopClosing::ComputeOptimizedPose(shared_ptr<Frame> pKF, Sim3 &Scr, shared_ptr<CalibHessian> Hcalib,
                                           Mat77 &H, float windowSize) {

        LOG(INFO) << "computing optimized pose" << endl;
        int TH_HIGH = 50;
        vector<shared_ptr<Frame>> activeFrames = fullSystem->GetActiveFrames();
        // make the idepth map
        memset(idepthMap, 0, sizeof(float) * wG[0] * hG[0]);

        VecVec2 activePixels;
        // NOTE these residuals are not locked!
        // tcw unused?
        //SE3 Tcw = currentKF->getPose();
        for (shared_ptr<Frame> fh: activeFrames) {
            if (fh == currentKF) continue;
            for (shared_ptr<Feature> feat: fh->features) {
                if (feat->status == Feature::FeatureStatus::VALID &&
                    feat->point->status == Point::PointStatus::ACTIVE) {

                    shared_ptr<PointHessian> ph = feat->point->mpPH;
                    if (ph->lastResiduals[0].first != 0 && ph->lastResiduals[0].second == ResState::IN) {
                        shared_ptr<PointFrameResidual> r = ph->lastResiduals[0].first;
                        if (r->target.lock() != currentKF->frameHessian) continue;

                        int u = r->centerProjectedTo[0] + 0.5f;
                        int v = r->centerProjectedTo[1] + 0.5f;

                        float new_idepth = r->centerProjectedTo[2];
                        idepthMap[u + wG[0] * v] = new_idepth;
                    }
                }
            }
        }

        // dilate idepth by 1.
        for (auto &px: activePixels) {
            int idx = int(px[1] * wG[0] + px[0]);
            float idep = idepthMap[idx];

            idepthMap[idx - 1] = idep;
            idepthMap[idx + 1] = idep;
            idepthMap[idx + wG[0]] = idep;
            idepthMap[idx - wG[0]] = idep;
            idepthMap[idx - 1 + wG[0]] = idep;
            idepthMap[idx - 1 - wG[0]] = idep;
            idepthMap[idx + 1 - wG[0]] = idep;
            idepthMap[idx + 1 + wG[0]] = idep;
        }

        // optimize the current Tcw
        currentKF->SetFeatureGrid();

        // vector<shared_ptr<Feature>> matchedFeatures;
        VecVec3 matchedPoints;
        VecVec3 matchedFeatures;
        VecVec2 matchedPixels;

        // find more matches in the local map of pKF
        vector<shared_ptr<Feature>> candidateFeatures;

        for (auto &feat: pKF->features) {
            if (feat->status == Feature::FeatureStatus::VALID &&
                feat->point->status != Point::PointStatus::OUTLIER) {
                candidateFeatures.push_back(feat);
            }
        }

        int nmatches = 0;
        Mat33 Ki;
        Ki << Hcalib->fxli(), 0, Hcalib->cxli(), 0, Hcalib->fyli(), Hcalib->cyli(), 0, 0, 1;

        // search by projection
        for (auto &p: candidateFeatures) {

            Vec3 pRef = (1.0 / p->invD) * Vec3(
                    Hcalib->fxli() * (p->uv[0] - Hcalib->cxl()),
                    Hcalib->fyli() * (p->uv[1] - Hcalib->cyl()),
                    1
            );
            Vec3 pc = Scr * pRef;

            float x = pc[0] / pc[2];
            float y = pc[1] / pc[2];
            float u = Hcalib->fxl() * x + Hcalib->cxl();
            float v = Hcalib->fyl() * y + Hcalib->cyl();

            int bestDist = 256;
            int bestDist2 = 256;
            int bestIdx = -1;

            // look for points nearby
            auto indices = currentKF->GetFeatureInGrid(u, v, windowSize);
            float idepth = 0;

            for (size_t &k: indices) {
                shared_ptr<Feature> &feat = currentKF->features[k];
                if (fabsf(feat->angle - p->angle) < 0.2) {
                    // check rotation first
                    int dist = FeatureMatcher::DescriptorDistance(feat->descriptor,
                                                                  p->descriptor);

                    int ui = int(feat->uv[0] + 0.5f), vi = int(feat->uv[1] + 0.5f);
                    idepth = idepthMap[vi * wG[0] + ui];

                    if (idepth == 0) {
                        // NOTE don't need this idepth =0 because we need to estimate the scale
                        // well in stereo case you can still do this
                        continue;
                    }

                    if (dist < bestDist) {
                        bestDist2 = bestDist;
                        bestDist = dist;
                        bestIdx = k;
                    } else if (dist < bestDist2) {
                        bestDist2 = dist;
                    }
                }
            }

            if (bestDist <= TH_HIGH) {
                auto bestFeat = currentKF->features[bestIdx];

                int ui = int(bestFeat->uv[0] + 0.5f), vi = int(bestFeat->uv[1] + 0.5f);
                idepth = idepthMap[vi * wG[0] + ui];

                Vec3 pcurr = (1.0f / idepth) * (Ki * Vec3(bestFeat->uv[0], bestFeat->uv[1], 1));
                matchedPoints.push_back(pRef);
                matchedFeatures.push_back(pcurr);

                matchedPixels.push_back(Vec2(bestFeat->uv[0], bestFeat->uv[1]));

                nmatches++;
            }
        }

        if (nmatches < 10) {
            LOG(INFO) << "local map matches not enough: " << nmatches << endl;
            return false;
        }

        // pose optimization, note there maybe some mismatches
        // NOTE seems like there are multiple solutions if just use 3d-3d point pairs
        // setup g2o and solve the problem
        g2o::SparseOptimizer optimizer;
        g2o::BlockSolverX::LinearSolverType *linearSolver;
        linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();
        g2o::BlockSolverX *solver_ptr = new g2o::BlockSolverX(linearSolver);
        g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

        optimizer.setAlgorithm(solver);
        optimizer.setVerbose(false);

        // current KF's pose
        VertexSim3 *vSim3 = new VertexSim3();
        vSim3->setId(0);
        vSim3->setEstimate(Scr);
        optimizer.addVertex(vSim3);

        float sigma = 1.0;
        float sigma2 = sigma * sigma;
        float infor = 1.0 / sigma2;
        float th = 5.991 * infor;

        vector<EdgePointSim3 *> edgesSim3;
        vector<EdgeProjectPoseOnlySim3 *> edgesProjection;
        for (size_t i = 0; i < matchedFeatures.size(); i++) {

            // EdgeProjectPoseOnlySim3 *eProj = new EdgeProjectPoseOnlySim3(Hcalib->mpCam, matchedPoints[i]);
            EdgePointSim3 *e3d = new EdgePointSim3(matchedPoints[i]);
            e3d->setId(i);
            e3d->setVertex(0, vSim3);

            Mat33 inforMat = infor * Matrix3d::Identity();
            e3d->setInformation(inforMat);    // TODO should not be identity.
            e3d->setMeasurement(matchedFeatures[i]);
            edgesSim3.push_back(e3d);

            g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
            rk->setDelta(th);
            e3d->setRobustKernel(rk);
            optimizer.addEdge(e3d);

            EdgeProjectPoseOnlySim3 *eProj = new EdgeProjectPoseOnlySim3(Hcalib->camera, matchedPoints[i]);
            eProj->setVertex(0, vSim3);
            eProj->setInformation(Mat22::Identity());
            eProj->setMeasurement(matchedPixels[i]);
            optimizer.addEdge(eProj);
            edgesProjection.push_back(eProj);
        }

        LOG(INFO) << "Start optimization";
        optimizer.initializeOptimization(0);
        optimizer.optimize(10);

        int inliers = 0, outliers = 0;
        for (auto &e : edgesSim3) {
            if (e->chi2() > th || e->chi2() < 1e-9 /* maybe some bug in g2o */ ) {
                e->setLevel(1);
                outliers++;
            } else {
                e->setRobustKernel(nullptr);
                inliers++;
            }
        }

        LOG(INFO) << "inliers: " << inliers << ", outliers: " << outliers << endl;

        if (inliers < 15) // reject
            return false;

        optimizer.initializeOptimization(0);
        optimizer.optimize(10);

        // decide the inliers
        Sim3 ScrOpti = vSim3->estimate();

        if (ScrOpti.scale() == Scr.scale() || std::isnan(ScrOpti.scale()) || ScrOpti.scale() < 0)  // optimization failed
            return false;

        Scr = ScrOpti;
        Eigen::Map<Mat77> hessianData(vSim3->hessianData());
        H = hessianData;

        return true;
    }
}
