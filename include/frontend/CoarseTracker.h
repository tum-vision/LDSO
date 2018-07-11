#pragma once
#ifndef LDSO_COARSE_TRACKER_H_
#define LDSO_COARSE_TRACKER_H_

#include "NumTypes.h"
#include "internal/OptimizationBackend/MatrixAccumulators.h"
#include "internal/Residuals.h"
#include "internal/FrameHessian.h"
#include "internal/CalibHessian.h"

using namespace ldso;
using namespace ldso::internal;

namespace ldso {

    // the tracker
    class CoarseTracker {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        // constuctor, allocate memory and compute the camera intrinsics pyramid
        CoarseTracker(int w, int h);

        ~CoarseTracker() {
            for (float *ptr : ptrToDelete)
                delete[] ptr;
            ptrToDelete.clear();
        }

        /**
         ** @brief track the new coming frame and estimate its pose and light parameters
         * @param[in] newFrameHessian the new frame
         * @param[in] lastToNew_out initial value of T_new_last
         * @param[out] aff_g2l_out affine transform
         * @param[in] coarsestLvl the first pyramid level (default=5)
         * @param[in] minResForAbort if residual > 1.5* minResForAbort, return false
         * @return true if track is good
         */
        bool trackNewestCoarse(
                shared_ptr<FrameHessian> newFrameHessian,
                SE3 &lastToNew_out, AffLight &aff_g2l_out,
                int coarsestLvl, Vec5 minResForAbort);

        void setCoarseTrackingRef(
                std::vector<shared_ptr<FrameHessian>>& frameHessians);

        /**
         * Create camera intrinsics buffer given the calibrated parameters
         * @param HCalib
         */
        void makeK(
                shared_ptr<CalibHessian> HCalib);

        shared_ptr<FrameHessian> lastRef = nullptr;     // the reference frame
        AffLight lastRef_aff_g2l;                       // affine light transform
        shared_ptr<FrameHessian> newFrame = nullptr;    // the new coming frame
        int refFrameID = -1;

        // act as pure ouptut
        Vec5 lastResiduals;
        Vec3 lastFlowIndicators;
        double firstCoarseRMSE = 0;

        // camera and image parameters in each pyramid
        Mat33f K[PYR_LEVELS];
        Mat33f Ki[PYR_LEVELS];
        float fx[PYR_LEVELS];
        float fy[PYR_LEVELS];
        float fxi[PYR_LEVELS];
        float fyi[PYR_LEVELS];
        float cx[PYR_LEVELS];
        float cy[PYR_LEVELS];
        float cxi[PYR_LEVELS];
        float cyi[PYR_LEVELS];
        int w[PYR_LEVELS];
        int h[PYR_LEVELS];

    private:
        void makeCoarseDepthL0(std::vector<shared_ptr<FrameHessian>> frameHessians);

        /**
         * @param[in] lvl the pyramid level
         * @param[in] refToNew pose from reference to current
         * @param[in] aff_g2l affine light transform from g to l
         * @param[in] cutoffTH cut off threshold, if residual > cutoffTH, then make residual = max energy. Similar with robust kernel in g2o.
         * @return the residual vector (a bit complicated, the the last lines in this func.)
         */
        Vec6 calcRes(int lvl, const SE3 &refToNew, AffLight aff_g2l, float cutoffTH);

        /**
         * @brief SSE accelerated Gauss-Newton
         * NOTE it uses some cache data in "warped buffers"
         * @param[in] lvl image pyramid level
         * @param[out] H_out Hessian matrix
         * @param[out] b_out bias vector
         * @param[in] refToNew Transform matrix from ref to new
         * @param[in] aff_g2l affine light transform
         */
        void calcGSSSE(int lvl, Mat88 &H_out, Vec8 &b_out, const SE3 &refToNew, AffLight aff_g2l);


        // point cloud buffers
        // wxh in each pyramid layer
        float *pc_u[PYR_LEVELS];            // u coordinates
        float *pc_v[PYR_LEVELS];            // v coordinates
        float *pc_idepth[PYR_LEVELS];       // inv depth in the reference
        float *pc_color[PYR_LEVELS];        // color of the reference patches
        int pc_n[PYR_LEVELS];               // number of points in each layer

        // warped buffer, used as wxh images
        float *buf_warped_idepth;
        float *buf_warped_u;
        float *buf_warped_v;
        float *buf_warped_dx;
        float *buf_warped_dy;
        float *buf_warped_residual;
        float *buf_warped_weight;
        float *buf_warped_refColor;
        int buf_warped_n;

        float *idepth[PYR_LEVELS];
        float *weightSums[PYR_LEVELS];
        float *weightSums_bak[PYR_LEVELS];

        std::vector<float *> ptrToDelete;    // all allocated memory, will be deleted in deconstructor
        Accumulator9 acc;
    };

    // the distance map
    class CoarseDistanceMap {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        CoarseDistanceMap(int w, int h);

        ~CoarseDistanceMap();

        void makeDistanceMap(
                std::vector<shared_ptr<FrameHessian>>& frameHessians,
                shared_ptr<FrameHessian> frame);

        void makeK(shared_ptr<CalibHessian> HCalib);

        float *fwdWarpedIDDistFinal;

        Mat33f K[PYR_LEVELS];
        Mat33f Ki[PYR_LEVELS];
        float fx[PYR_LEVELS];
        float fy[PYR_LEVELS];
        float fxi[PYR_LEVELS];
        float fyi[PYR_LEVELS];
        float cx[PYR_LEVELS];
        float cy[PYR_LEVELS];
        float cxi[PYR_LEVELS];
        float cyi[PYR_LEVELS];
        int w[PYR_LEVELS];
        int h[PYR_LEVELS];

        void addIntoDistFinal(int u, int v);

    private:

        PointFrameResidual **coarseProjectionGrid;
        int *coarseProjectionGridNum;
        Eigen::Vector2i *bfsList1;
        Eigen::Vector2i *bfsList2;

        void growDistBFS(int bfsNum);
    };
}

#endif // LDSO_COARSE_TRACKER_H_
