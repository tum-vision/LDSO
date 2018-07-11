#pragma once
#ifndef LDSO_FEATURE_MATCHER_H_
#define LDSO_FEATURE_MATCHER_H_

#include "NumTypes.h"
#include "Frame.h"
#include "Map.h"

#include "internal/CalibHessian.h"
#include "internal/OptimizationBackend/EnergyFunctional.h"

using namespace ldso::internal;

namespace ldso {

    /**
     * Match structure
     */
    struct Match {
        Match(int _index1 = -1, int _index2 = -1, int _dist = -1) : index1(_index1), index2(_index2), dist(_dist) {}

        int index1 = -1;
        int index2 = -1;
        int dist = -1;
    };

    class FeatureMatcher {

    public:
        FeatureMatcher(float nnRatio = 0.6, bool checkRot = true) :
                nnRatio(nnRatio), checkOrientation(checkRot) {}

        /// the distance of two descriptors
        static int DescriptorDistance(const unsigned char *desc1, const unsigned char *desc2);

        /**
         * Brute-force Search for feature matching
         * @param frame1
         * @param frame2
         * @param matches
         * @return
         */
        int SearchBruteForce(shared_ptr<Frame> frame1, shared_ptr<Frame> frame2, std::vector<Match> &matches);

        /**
         * Search by bag-of-words model
         * @param frame1
         * @param frame2
         * @param matches
         * @return
         */
        int SearchByBoW(shared_ptr<Frame> frame1, shared_ptr<Frame> frame2, std::vector<Match> &matches);

        // draw matches, will block until user press a key, return the cv::waitkey code
        int DrawMatches( shared_ptr<Frame> frame1, shared_ptr<Frame> frame2, std::vector<Match>& matches );

    private:
        float nnRatio = 0.6;
        bool checkOrientation = true;

        // configuation
        const int TH_LOW = 50;
        const int TH_HIGH = 100;
        const int HISTO_LENGTH = 30;

    };
}

#endif