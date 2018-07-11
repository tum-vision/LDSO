#pragma once
#ifndef LDSO_FEATURE_H_
#define LDSO_FEATURE_H_

#include <memory>

using namespace std;

#include "NumTypes.h"

namespace ldso {

    struct Frame;
    struct Point;

    namespace internal {
        class ImmaturePoint;
    }

    /**
     * Feature is a 2D point in the image. A triangulated feature will have an associated 3D map point, but an immature
     * feature will not (instead it has a immature point). You can access a feature's host frame and the map point.
     *
     * Feature may have a descriptor (ORB currently) if it is a corner (with isCorner == true),
     * otherwise the descriptor, angle and level are always kept as the default value. Described features can be used
     * for feature matching, loop closing and bag-of-words ... anything you expect in a feature-based SLAM.
     *
     * NOTE outlier features will also be kept in frame know. If you worry about the memory cost you can just clean them
     */

    struct Feature {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        /**
         * Feature status
         */
        enum FeatureStatus {
            IMMATURE = 0,    // if immature, the ip will have the immature data
            VALID,           // have a valid map point (but the point may also be outdated or margined ... )
            OUTLIER          // the immature point diverges, or the map point becomes an outlier
        };  // the status of feature

        Feature(float u, float v, shared_ptr<Frame> host) : host(host), uv(Vec2f(u, v)) {}

        ~Feature() {}

        /**
         * Create a map point from the immature structure
         */
        void CreateFromImmature();

        /**
         * release the internal immature point data
         */
        void ReleaseImmature();

        /**
         * release the map point
         */
        void ReleaseMapPoint();

        /**
         * release all internal data
         */
        inline void ReleaseAll() {
            ReleaseImmature();
            ReleaseMapPoint();
        }

        // save and load
        void save(ofstream &fout);

        void load(ifstream &fin, vector<shared_ptr<Frame>> &allKFs);

        // =====================================================================================================
        FeatureStatus status = IMMATURE;   // status of this feature

        weak_ptr<Frame> host;   // the host frame

        Vec2f uv = Vec2f(0, 0);               // pixel position in image
        float invD = -1;                  // inverse depth, invalid if < 0, computed by dso's sliding window
        shared_ptr<Point> point = nullptr;    // corresponding 3D point, nullptr if it is an immature point

        // feature stuffs
        float angle = 0;        // rotation
        float score = 0;        // shi-tomasi score
        bool isCorner = false; // indicating if this is a corner
        int level = 0;         // which pyramid level is the feature computed
        unsigned char descriptor[32] = {0};  // ORB descriptors

        // internal structures for optimizing immature points
        shared_ptr<internal::ImmaturePoint> ip = nullptr;  // the immature point
    };
}

#endif