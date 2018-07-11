#pragma once
#ifndef LDSO_RESIDUALS_H_
#define LDSO_RESIDUALS_H_

#include <memory>

using namespace std;

#include "NumTypes.h"
#include "internal/RawResidualJacobian.h"

namespace ldso {

    namespace internal {

        /**
         * photometric residuals defined in DSO
         */
        class PointHessian;

        class FrameHessian;

        class CalibHessian;

        class EnergyFunctional;

        enum ResLocation {
            ACTIVE = 0, LINEARIZED, MARGINALIZED, NONE
        };

        enum ResState {
            IN = 0, OOB, OUTLIER
        }; // Residual state: inside, outside, outlier

        struct FullJacRowT {
            Eigen::Vector2f projectedTo[MAX_RES_PER_POINT];
        };

        // Photometric reprojection Error
        class PointFrameResidual {
        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

            PointFrameResidual() : J(new RawResidualJacobian) {}

            PointFrameResidual(shared_ptr<PointHessian> point_, shared_ptr<FrameHessian> host_,
                               shared_ptr<FrameHessian> target_) : J(new RawResidualJacobian) {
                point = point_;
                host = host_;
                target = target_;
                resetOOB();
            }

            /**
             * linearize the reprojection, create jacobian matrices
             * @param HCalib
             * @return the new energy
             */
            virtual double linearize(shared_ptr<CalibHessian> &HCalib);

            virtual void resetOOB() {
                state_NewEnergy = state_energy = 0;
                state_NewState = ResState::OUTLIER;
                setState(ResState::IN);
            };

            // 将state_NewState的状态更新至当前状态
            void applyRes(bool copyJacobians) {

                if (copyJacobians) {
                    if (state_state == ResState::OOB) {
                        return;
                    }

                    if (state_NewState == ResState::IN) {
                        isActiveAndIsGoodNEW = true;
                        takeData();
                    } else {
                        isActiveAndIsGoodNEW = false;
                    }
                }

                state_state = state_NewState;
                state_energy = state_NewEnergy;
            }

            void setState(ResState s) { state_state = s; }

            static int instanceCounter;
            ResState state_state = ResState::OUTLIER;
            double state_energy = 0;
            ResState state_NewState = ResState::OUTLIER;
            double state_NewEnergy = 0;
            double state_NewEnergyWithOutlier = 0;

            weak_ptr<PointHessian> point;
            weak_ptr<FrameHessian> host;
            weak_ptr<FrameHessian> target;
            shared_ptr<RawResidualJacobian> J = nullptr;

            bool isNew = true;
            Eigen::Vector2f projectedTo[MAX_RES_PER_POINT]; // 从host到target的投影点
            Vec3f centerProjectedTo;

            // ==================================================================================== //
            // Energy stuffs
            inline bool isActive() const { return isActiveAndIsGoodNEW; }

            // fix the jacobians
            void fixLinearizationF(shared_ptr<EnergyFunctional> ef);

            int hostIDX = 0, targetIDX = 0;

            VecNRf res_toZeroF;
            Vec8f JpJdF = Vec8f::Zero();
            bool isLinearized = false;  // if linearization is fixed.

            // if residual is not OOB & not OUTLIER & should be used during accumulations
            bool isActiveAndIsGoodNEW = false;

            void takeData() {
                Vec2f JI_JI_Jd = J->JIdx2 * J->Jpdd;
                for (int i = 0; i < 6; i++)
                    JpJdF[i] = J->Jpdxi[0][i] * JI_JI_Jd[0] + J->Jpdxi[1][i] * JI_JI_Jd[1];
                JpJdF.segment<2>(6) = J->JabJIdx * J->Jpdd;
            }

        };
    }
}

#endif // LDSO_RESIDUALS_H_
