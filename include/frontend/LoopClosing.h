#pragma once
#ifndef LDSO_LOOP_CLOSING_H_
#define LDSO_LOOP_CLOSING_H_

#include "NumTypes.h"
#include "Frame.h"
#include "Map.h"
#include "FeatureMatcher.h"
#include "CoarseTracker.h"

#include "internal/CalibHessian.h"

#include <list>
#include <queue>
#include <mutex>

using namespace std;

using ldso::internal::CalibHessian;

namespace ldso {
    class FullSystem;

    /**
     * Loop closing thread, also used for correcting loops
     *
     * loop closing is running in a single thread, receiving new keyframes from the front end. It will seprate all keyframes into several "consistent groups", and if the newest keyframe seems to be consistent with a previous group, we say a loop closure is detected. And once we find a loop, we will check the sim3 and correct it using a global bundle adjustment.
     */
    class LoopClosing {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        // Consistent group, the first is a group of keyframes that are considered as consistent, and the second is how many times they are detected
        typedef pair<set<shared_ptr<Frame>>, int> ConsistentGroup;

        LoopClosing(FullSystem *fullSystem);

        ~LoopClosing() {
            if (idepthMap) delete[] idepthMap;
        }

        void InsertKeyFrame(shared_ptr<Frame> &frame);

        /**
         * detect loop candidates from the keyframe database
         * @param frame
         * @return true if there is at least one loop candidate
         */
        bool DetectLoop(shared_ptr<Frame> &frame);

        /**
         * compute RANSAC pnp in loop frames
         * however this function will not try to optimize the pose graph, which will be done in full system
         * @return true if find enough inliers
         */
        bool CorrectLoop(shared_ptr<CalibHessian> Hcalib);

        void Run();

        /**
         * set main loop to finish
         * @param finish
         */
        void SetFinish(bool finish = true) {

            needFinish = finish;
            LOG(INFO) << "wait loop closing to join" << endl;
            mainLoop.join();
            while (globalMap && globalMap->Idle() == false) {
                usleep(10000);
            }

            if (needPoseGraph) {
                if (globalMap) {
                    globalMap->OptimizeALLKFs();
                    usleep(5000);
                }
                while (globalMap && globalMap->Idle() == false) {
                    usleep(10000);
                }
            }
            LOG(INFO) << "Loop closing thread is finished" << endl;
        }

    private:
        /**
         * compute an optimized sim3 from a given keyframe to current frame
         * @param pKF given keyframe, also loop candidate
         * @param Scr Sim(3) from ref to current
         * @param Hcalib camera intrinsics
         * @param windowSize projection window size
         * @param hessian optimized hessian matrix (based on geometric error)
         * @return true if computation successes
         */
        bool
        ComputeOptimizedPose(shared_ptr<Frame> pKF, Sim3 &Scr, shared_ptr<CalibHessian> Hcalib, Mat77 &hessian,
                             float windowSize = 5.0);

        // data
        FullSystem *fullSystem;
        shared_ptr<Map> globalMap = nullptr;  // global map

        // shared_ptr<KeyFrameDatabase> kfDB = nullptr;
        shared_ptr<DBoW3::Database> kfDB = nullptr;
        shared_ptr<ORBVocabulary> voc = nullptr;

        shared_ptr<Frame> candidateKF = nullptr;
        vector<shared_ptr<Frame>> allKF;
        map<DBoW3::EntryId, shared_ptr<Frame>> checkedKFs;    // keyframes that are recorded.
        int maxKFId = 0;
        shared_ptr<Frame> currentKF = nullptr;

        // loop kf queue
        deque<shared_ptr<Frame>> KFqueue;
        mutex mutexKFQueue;
        shared_ptr<CoarseDistanceMap> coarseDistanceMap = nullptr;  // Need distance map to correct the sim3 error
        bool finished = false;
        shared_ptr<CalibHessian> Hcalib = nullptr;
        bool needFinish = false;
        bool needPoseGraph = false;
        float *idepthMap = nullptr;   // i hate this float*
        thread mainLoop;

        // parameters
        double minScoreAccept = 0.06;
        int kfGap = 10;

    };
}

#endif // LDSO_LOOP_CLOSING_H_
