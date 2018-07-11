#pragma once
#ifndef LDSO_PIXEl_SELECTOR_H_
#define LDSO_PIXEl_SELECTOR_H_

#include "NumTypes.h"
#include "Settings.h"
#include "Frame.h"

using namespace ldso;
using ldso::internal::FrameHessian;

namespace ldso {

    enum PixelSelectorStatus {
        PIXSEL_VOID = 0, PIXSEL_1, PIXSEL_2, PIXSEL_3
    };

    class PixelSelector {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        /**
         * select points from an input frame
         * @param fh
         * @param map_out an image of point selection, same size with the input image and value != 0 means the point is selected
         * // below is some parameters affecting the selection
         * @param density
         * @param recursionsLeft
         * @param plot
         * @param thFactor
         * @return number of selected points
         */
        int makeMaps(
                const shared_ptr<FrameHessian> fh,
                float *map_out, float density, int recursionsLeft = 1, bool plot = false, float thFactor = 1);

        PixelSelector(int w, int h);

        ~PixelSelector();

        void makeHists(shared_ptr<FrameHessian> fh);

        int currentPotential;
        bool allowFast = false;
    private:

        Eigen::Vector3i select(const shared_ptr<FrameHessian> fh,
                               float *map_out, int pot, float thFactor = 1);


        unsigned char *randomPattern = nullptr;

        int *gradHist = nullptr;
        float *ths = nullptr;
        float *thsSmoothed = nullptr;
        int thsStep = 0;
        shared_ptr<FrameHessian> gradHistFrame = nullptr;
    };

    // from pixel selector.h
    const float minUseGrad_pixsel = 10;

    template<int pot>
    inline int gridMaxSelection(Eigen::Vector3f *grads, bool *map_out, int w, int h, float THFac) {

        memset(map_out, 0, sizeof(bool) * w * h);

        int numGood = 0;
        for (int y = 1; y < h - pot; y += pot) {
            for (int x = 1; x < w - pot; x += pot) {
                int bestXXID = -1;
                int bestYYID = -1;
                int bestXYID = -1;
                int bestYXID = -1;

                float bestXX = 0, bestYY = 0, bestXY = 0, bestYX = 0;

                Eigen::Vector3f *grads0 = grads + x + y * w;
                for (int dx = 0; dx < pot; dx++)
                    for (int dy = 0; dy < pot; dy++) {
                        int idx = dx + dy * w;
                        Eigen::Vector3f g = grads0[idx];
                        float sqgd = g.tail<2>().squaredNorm();
                        float TH = THFac * minUseGrad_pixsel * (0.75f);

                        if (sqgd > TH * TH) {
                            float agx = fabs((float) g[1]);
                            if (agx > bestXX) {
                                bestXX = agx;
                                bestXXID = idx;
                            }

                            float agy = fabs((float) g[2]);
                            if (agy > bestYY) {
                                bestYY = agy;
                                bestYYID = idx;
                            }

                            float gxpy = fabs((float) (g[1] - g[2]));
                            if (gxpy > bestXY) {
                                bestXY = gxpy;
                                bestXYID = idx;
                            }

                            float gxmy = fabs((float) (g[1] + g[2]));
                            if (gxmy > bestYX) {
                                bestYX = gxmy;
                                bestYXID = idx;
                            }
                        }
                    }

                bool *map0 = map_out + x + y * w;

                if (bestXXID >= 0) {
                    if (!map0[bestXXID])
                        numGood++;
                    map0[bestXXID] = true;

                }
                if (bestYYID >= 0) {
                    if (!map0[bestYYID])
                        numGood++;
                    map0[bestYYID] = true;

                }
                if (bestXYID >= 0) {
                    if (!map0[bestXYID])
                        numGood++;
                    map0[bestXYID] = true;

                }
                if (bestYXID >= 0) {
                    if (!map0[bestYXID])
                        numGood++;
                    map0[bestYXID] = true;

                }
            }
        }

        return numGood;
    }


    inline int gridMaxSelection(Eigen::Vector3f *grads, bool *map_out, int w, int h, int pot, float THFac) {

        memset(map_out, 0, sizeof(bool) * w * h);

        int numGood = 0;
        for (int y = 1; y < h - pot; y += pot) {
            for (int x = 1; x < w - pot; x += pot) {
                int bestXXID = -1;
                int bestYYID = -1;
                int bestXYID = -1;
                int bestYXID = -1;

                float bestXX = 0, bestYY = 0, bestXY = 0, bestYX = 0;

                Eigen::Vector3f *grads0 = grads + x + y * w;
                for (int dx = 0; dx < pot; dx++)
                    for (int dy = 0; dy < pot; dy++) {
                        int idx = dx + dy * w;
                        Eigen::Vector3f g = grads0[idx];
                        float sqgd = g.tail<2>().squaredNorm();
                        float TH = THFac * minUseGrad_pixsel * (0.75f);

                        if (sqgd > TH * TH) {
                            float agx = fabs((float) g[1]);
                            if (agx > bestXX) {
                                bestXX = agx;
                                bestXXID = idx;
                            }

                            float agy = fabs((float) g[2]);
                            if (agy > bestYY) {
                                bestYY = agy;
                                bestYYID = idx;
                            }

                            float gxpy = fabs((float) (g[1] - g[2]));
                            if (gxpy > bestXY) {
                                bestXY = gxpy;
                                bestXYID = idx;
                            }

                            float gxmy = fabs((float) (g[1] + g[2]));
                            if (gxmy > bestYX) {
                                bestYX = gxmy;
                                bestYXID = idx;
                            }
                        }
                    }

                bool *map0 = map_out + x + y * w;

                if (bestXXID >= 0) {
                    if (!map0[bestXXID])
                        numGood++;
                    map0[bestXXID] = true;

                }
                if (bestYYID >= 0) {
                    if (!map0[bestYYID])
                        numGood++;
                    map0[bestYYID] = true;

                }
                if (bestXYID >= 0) {
                    if (!map0[bestXYID])
                        numGood++;
                    map0[bestXYID] = true;

                }
                if (bestYXID >= 0) {
                    if (!map0[bestYXID])
                        numGood++;
                    map0[bestYXID] = true;

                }
            }
        }

        return numGood;
    }


    inline int makePixelStatus(Eigen::Vector3f *grads, bool *map, int w, int h, float desiredDensity, int recsLeft = 5,
                               float THFac = 1) {
        if (sparsityFactor < 1) sparsityFactor = 1;

        int numGoodPoints;


        if (sparsityFactor == 1) numGoodPoints = gridMaxSelection<1>(grads, map, w, h, THFac);
        else if (sparsityFactor == 2) numGoodPoints = gridMaxSelection<2>(grads, map, w, h, THFac);
        else if (sparsityFactor == 3) numGoodPoints = gridMaxSelection<3>(grads, map, w, h, THFac);
        else if (sparsityFactor == 4) numGoodPoints = gridMaxSelection<4>(grads, map, w, h, THFac);
        else if (sparsityFactor == 5) numGoodPoints = gridMaxSelection<5>(grads, map, w, h, THFac);
        else if (sparsityFactor == 6) numGoodPoints = gridMaxSelection<6>(grads, map, w, h, THFac);
        else if (sparsityFactor == 7) numGoodPoints = gridMaxSelection<7>(grads, map, w, h, THFac);
        else if (sparsityFactor == 8) numGoodPoints = gridMaxSelection<8>(grads, map, w, h, THFac);
        else if (sparsityFactor == 9) numGoodPoints = gridMaxSelection<9>(grads, map, w, h, THFac);
        else if (sparsityFactor == 10) numGoodPoints = gridMaxSelection<10>(grads, map, w, h, THFac);
        else if (sparsityFactor == 11) numGoodPoints = gridMaxSelection<11>(grads, map, w, h, THFac);
        else numGoodPoints = gridMaxSelection(grads, map, w, h, sparsityFactor, THFac);


        /*
         * #points is approximately proportional to sparsityFactor^2.
         */

        float quotia = numGoodPoints / (float) (desiredDensity);

        int newSparsity = (sparsityFactor * sqrtf(quotia)) + 0.7f;


        if (newSparsity < 1) newSparsity = 1;


        float oldTHFac = THFac;
        if (newSparsity == 1 && sparsityFactor == 1) THFac = 0.5;


        if ((abs(newSparsity - sparsityFactor) < 1 && THFac == oldTHFac) ||
            (quotia > 0.8 && 1.0f / quotia > 0.8) ||
            recsLeft == 0) {

            //all good
            sparsityFactor = newSparsity;
            return numGoodPoints;
        } else {
            // re-evaluate.
            sparsityFactor = newSparsity;
            return makePixelStatus(grads, map, w, h, desiredDensity, recsLeft - 1, THFac);
        }
    }
}
#endif // LDSO_PIXEl_SELECTOR_H_
