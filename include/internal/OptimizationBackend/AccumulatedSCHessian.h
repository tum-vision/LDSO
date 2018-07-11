#pragma once
#ifndef LDSO_ACCUMULATED_SC_HESSIAN_H_
#define LDSO_ACCUMULATED_SC_HESSIAN_H_

#include "NumTypes.h"
#include "Settings.h"
#include "internal/OptimizationBackend/MatrixAccumulators.h"
#include "internal/IndexThreadReduce.h"

namespace ldso {

    namespace internal {

        class EnergyFunctional;
        class PointHessian;

        class AccumulatedSCHessianSSE {
        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

            inline AccumulatedSCHessianSSE() {
                for (int i = 0; i < NUM_THREADS; i++) {
                    accE[i] = 0;
                    accEB[i] = 0;
                    accD[i] = 0;
                    nframes[i] = 0;
                }
            };

            inline ~AccumulatedSCHessianSSE() {
                for (int i = 0; i < NUM_THREADS; i++) {
                    if (accE[i] != 0) delete[] accE[i];
                    if (accEB[i] != 0) delete[] accEB[i];
                    if (accD[i] != 0) delete[] accD[i];
                }
            };

            inline void setZero(int n, int min = 0, int max = 1, Vec10 *stats = 0, int tid = 0) {
                if (n != nframes[tid]) {
                    if (accE[tid] != 0) delete[] accE[tid];
                    if (accEB[tid] != 0) delete[] accEB[tid];
                    if (accD[tid] != 0) delete[] accD[tid];
                    accE[tid] = new AccumulatorXX<8, CPARS>[n * n];
                    accEB[tid] = new AccumulatorX<8>[n * n];
                    accD[tid] = new AccumulatorXX<8, 8>[n * n * n];
                }
                accbc[tid].initialize();
                accHcc[tid].initialize();

                for (int i = 0; i < n * n; i++) {
                    accE[tid][i].initialize();
                    accEB[tid][i].initialize();

                    for (int j = 0; j < n; j++)
                        accD[tid][i * n + j].initialize();
                }
                nframes[tid] = n;
            }

            void stitchDouble(MatXX &H_sc, VecX &b_sc, const EnergyFunctional * const, int tid = 0);

            void addPoint( shared_ptr<PointHessian> p, bool shiftPriorToZero, int tid = 0);

            void stitchDoubleMT(IndexThreadReduce<Vec10>* red, MatXX &H, VecX &b, EnergyFunctional const *const EF,
                                bool MT) {
                // sum up, splitting by bock in square.
                if (MT) {
                    MatXX Hs[NUM_THREADS];
                    VecX bs[NUM_THREADS];
                    for (int i = 0; i < NUM_THREADS; i++) {
                        assert(nframes[0] == nframes[i]);
                        Hs[i] = MatXX::Zero(nframes[0] * 8 + CPARS, nframes[0] * 8 + CPARS);
                        bs[i] = VecX::Zero(nframes[0] * 8 + CPARS);
                    }

                    red->reduce(std::bind(&AccumulatedSCHessianSSE::stitchDoubleInternal,
                                          this, Hs, bs, EF, _1, _2, _3, _4), 0, nframes[0] * nframes[0], 0);

                    // sum up results
                    H = Hs[0];
                    b = bs[0];

                    for (int i = 1; i < NUM_THREADS; i++) {
                        H.noalias() += Hs[i];
                        b.noalias() += bs[i];
                    }
                } else {
                    H = MatXX::Zero(nframes[0] * 8 + CPARS, nframes[0] * 8 + CPARS);
                    b = VecX::Zero(nframes[0] * 8 + CPARS);
                    stitchDoubleInternal(&H, &b, EF, 0, nframes[0] * nframes[0], 0, -1);
                }

                // make diagonal by copying over parts.
                for (int h = 0; h < nframes[0]; h++) {
                    int hIdx = CPARS + h * 8;
                    H.block<CPARS, 8>(0, hIdx).noalias() = H.block<8, CPARS>(hIdx, 0).transpose();
                }
            }

            AccumulatorXX<8, CPARS> *accE[NUM_THREADS];
            AccumulatorX<8> *accEB[NUM_THREADS];
            AccumulatorXX<8, 8> *accD[NUM_THREADS];
            AccumulatorXX<CPARS, CPARS> accHcc[NUM_THREADS];
            AccumulatorX<CPARS> accbc[NUM_THREADS];
            int nframes[NUM_THREADS];

            void addPointsInternal(
                    std::vector<shared_ptr<PointHessian>> *points, bool shiftPriorToZero,
                    int min = 0, int max = 1, Vec10 *stats = 0, int tid = 0) {
                for (int i = min; i < max; i++) addPoint((*points)[i], shiftPriorToZero, tid);
            }

        private:

            void stitchDoubleInternal(
                    MatXX *H, VecX *b, EnergyFunctional const *const EF,
                    int min, int max, Vec10 *stats, int tid);
        };

    }
}

#endif // LDSO_ACCUMULATED_SC_HESSIAN_H_
