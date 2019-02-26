#pragma once
#ifndef LDSO_FEATURE_DETECTOR_H_
#define LDSO_FEATURE_DETECTOR_H_

#include "NumTypes.h"
#include "Frame.h"

#include "internal/GlobalCalib.h"
#include "internal/FrameHessian.h"

#include <cmath>

using namespace ldso::internal;

namespace ldso {

    class FeatureDetector {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        const int HALF_PATCH_SIZE = 15; // half patch size for computing ORB descriptor

        FeatureDetector();

        ~FeatureDetector();

        /**
         * detector corners
         * @param frame input frame, feature should already be created by PixelSelector
         * @return number of selected features
         */
        int DetectCorners(int nFeatures, shared_ptr<Frame> &frame);

        int ComputeDescriptor(shared_ptr<Frame> &frame, shared_ptr<Feature> feat);

        /**
         * debug stuffs
         */
        void DrawFeatures(shared_ptr<Frame> &frame, const string &windowName = "corner");

    private:
        /**
         * shi-tomasi score
         * @param frame, must have frame hessian since we need dI
         * @param u
         * @param v
         * @param halfbox
         * @return
         */
        inline float
        ShiTomasiScore(shared_ptr<Frame> &frame, const float &u, const float &v, int halfbox = 4, int level = 0) {

            float dXX = 0.0;
            float dYY = 0.0;
            float dXY = 0.0;

            const int box_size = 2 * halfbox;
            const int box_area = box_size * box_size;
            const int x_min = u - halfbox;
            const int x_max = u + halfbox;
            const int y_min = v - halfbox;
            const int y_max = v + halfbox;

            if (x_min < 1 || x_max >= wG[level] - 1 || y_min < 1 || y_max >= hG[level] - 1)
                return 0.0; // patch is too close to the boundary
            const int stride = wG[level];

            for (int y = y_min; y < y_max; ++y) {
                for (int x = x_min; x < x_max; ++x) {
                    float dx = frame->frameHessian->dIp[level][y * stride + x][1];
                    float dy = frame->frameHessian->dIp[level][y * stride + x][2];
                    dXX += dx * dx;
                    dYY += dy * dy;
                    dXY += dx * dy;
                }
            }

            // Find and return smaller eigenvalue:
            dXX = dXX / (2.0 * box_area);
            dYY = dYY / (2.0 * box_area);
            dXY = dXY / (2.0 * box_area);
            return 0.5 * (dXX + dYY - sqrt((dXX + dYY) * (dXX + dYY) - 4 * (dXX * dYY - dXY * dXY)));
        }

        /**
         * compute the rotation of a feature point
         * @param image the image stored in FrameHessian
         * @param pt keypoint position
         * @param u_max
         * @return
         */
        inline float IC_Angle(const Vec3f *image, const Vec2f &pt, int level = 0) {

            float m_01 = 0, m_10 = 0;
            const Vec3f *center = image + int(pt[1]) * wG[level] + int(pt[0]);

            // Treat the center line differently, v=0
            for (int u = -HALF_PATCH_SIZE; u <= HALF_PATCH_SIZE; ++u)
                m_10 += u * center[u][0];

            // Go line by line in the circular patch
            int step = wG[level];
            for (int v = 1; v <= HALF_PATCH_SIZE; ++v) {
                // Proceed over the two lines
                float v_sum = 0;
                int d = umax[v];
                for (int u = -d; u <= d; ++u) {
                    float val_plus = center[u + v * step][0], val_minus = center[u - v * step][0];
                    v_sum += (val_plus - val_minus);
                    m_10 += u * (val_plus + val_minus);
                }
                m_01 += v * v_sum;
            }
            return atan2f(m_01, m_10);
        }

        // configurations
        // unused?
        //float minScoreTH = 0.05;
        //float minDistance = 10;

        // static data
        std::vector<int> umax;  // used to compute rotation
    };
}

#endif // LDSO_FEATURE_DETECTOR_H_
