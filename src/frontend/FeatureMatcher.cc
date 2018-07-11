#include "Feature.h"
#include "frontend/FeatureMatcher.h"
#include "internal/GlobalCalib.h"
#include "internal/FrameHessian.h"
#include "internal/PointHessian.h"
#include "internal/ResidualProjections.h"
#include "internal/Residuals.h"

#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>

using namespace ldso::internal;

namespace ldso {

    int FeatureMatcher::DescriptorDistance(const unsigned char *desc1, const unsigned char *desc2) {

        int dist = 0;
        const int *pa = (int *) desc1;
        const int *pb = (int *) desc2;

        for (int i = 0; i < 8; i++, pa++, pb++) {
            unsigned int v = *pa ^*pb;
#ifdef __SSE2__
            dist += _mm_popcnt_u64(v);  // SSE is so easy
#else
            v = v - ( ( v >> 1 ) & 0x55555555 );
            v = ( v & 0x33333333 ) + ( ( v >> 2 ) & 0x33333333 );
            dist += ( ( ( v + ( v >> 4 ) ) & 0xF0F0F0F ) * 0x1010101 ) >> 24;
#endif
        }
        return dist;
    }

    int FeatureMatcher::SearchBruteForce(shared_ptr<Frame> frame1, shared_ptr<Frame> frame2,
                                         std::vector<Match> &matches) {

        matches.reserve(frame1->features.size());

        for (size_t i = 0; i < frame1->features.size(); i++) {
            shared_ptr<Feature> f1 = frame1->features[i];
            if (f1->isCorner == false)
                continue;
            int min_dist = 9999;
            int min_dist_index = -1;

            for (size_t j = 0; j < frame2->features.size(); j++) {
                shared_ptr<Feature> f2 = frame2->features[j];
                if (f2->isCorner == false)
                    continue;
                int dist = DescriptorDistance(f1->descriptor, f2->descriptor);
                if (dist < min_dist) {
                    min_dist = dist;
                    min_dist_index = j;
                }
            }

            if (min_dist < TH_LOW) {
                matches.push_back(Match(i, min_dist_index, min_dist));
            }
        }

        return matches.size();
    }

    int FeatureMatcher::SearchByBoW(shared_ptr<Frame> frame1, shared_ptr<Frame> frame2, std::vector<Match> &matches) {

        int nmatches = 0;
        matches.reserve(frame1->features.size());

        DBoW3::FeatureVector::const_iterator f1it = frame1->featVec.begin();
        DBoW3::FeatureVector::const_iterator f2it = frame2->featVec.begin();
        DBoW3::FeatureVector::const_iterator f1end = frame1->featVec.end();
        DBoW3::FeatureVector::const_iterator f2end = frame2->featVec.end();

        while (f1it != f1end && f2it != f2end) {
            if (f1it->first == f2it->first) {
                // from the same word
                const vector<unsigned int> vIdx1 = f1it->second;
                const vector<unsigned int> vIdx2 = f2it->second;

                for (auto &idx1: vIdx1) {
                    auto &feat1 = frame1->features[frame1->bowIdx[idx1]];
                    int bestDist1 = 256;    // 最近的
                    int bestIdx2 = -1;
                    int bestDist2 = 256;    // 第二近的

                    for (auto &idx2:vIdx2) {
                        auto &feat2 = frame2->features[frame2->bowIdx[idx2]];
                        int dist = DescriptorDistance(feat1->descriptor, feat2->descriptor);
                        if (dist < bestDist1) {
                            bestDist2 = bestDist1;
                            bestDist1 = dist;
                            bestIdx2 = frame2->bowIdx[idx2];
                        } else if (dist < bestDist2) {
                            bestDist2 = dist;
                        }
                    }

                    if (bestDist1 <= TH_LOW) {
                        if (static_cast<float>(bestDist1) < nnRatio * static_cast<float>(bestDist2)) {
                            // NN ratio
                            Match m;
                            m.index1 = frame1->bowIdx[idx1];
                            m.index2 = bestIdx2;
                            m.dist = bestDist1;
                            matches.push_back(m);
                            nmatches++;
                        }
                    }
                }

                f1it++;
                f2it++;

            } else if (f1it->first < f2it->first) {
                f1it = frame1->featVec.lower_bound(f2it->first);
            } else {
                f2it = frame2->featVec.lower_bound(f1it->first);
            }
        }

        return nmatches;
    }

    int FeatureMatcher::DrawMatches(shared_ptr<Frame> f1, shared_ptr<Frame> f2, std::vector<Match> &matches) {

        cv::Mat img(hG[0], wG[0] * 2, CV_8UC3);   // color image displayed
        f1->imgDisplay.copyTo(img(cv::Rect(0, 0, wG[0], hG[0])));
        f2->imgDisplay.copyTo(img(cv::Rect(wG[0], 0, wG[0], hG[0])));

        for (auto &m:matches) {
            cv::circle(img, cv::Point2f(f1->features[m.index1]->uv[0], f1->features[m.index1]->uv[1]), 1,
                       cv::Scalar(0, 250, 0), 2);

            cv::circle(img, cv::Point2f(f2->features[m.index2]->uv[0] + wG[0], f2->features[m.index2]->uv[1]), 1,
                       cv::Scalar(0, 250, 0), 2);

            cv::line(img, cv::Point2f(f1->features[m.index1]->uv[0], f1->features[m.index1]->uv[1]),
                     cv::Point2f(f2->features[m.index2]->uv[0] + wG[0], f2->features[m.index2]->uv[1]),
                     cv::Scalar(0, 250, 0),
                     1);
        }

        for (auto &feat: f1->features) {
            if (feat->isCorner)
                cv::circle(img, cv::Point2f(feat->uv[0], feat->uv[1]), 1, cv::Scalar(0, 0, 250), 2);
        }
        for (auto &feat: f2->features) {
            if (feat->isCorner)
                cv::circle(img, cv::Point2f(feat->uv[0] + wG[0], feat->uv[1]), 1, cv::Scalar(0, 0, 250), 2);
        }

        cv::imshow("Matches", img);
        return cv::waitKey(0);
    }
}