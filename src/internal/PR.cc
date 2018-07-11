#include "internal/PR.h"
#include "internal/GlobalCalib.h"
#include "internal/ResidualProjections.h"

using namespace ldso::internal;

namespace ldso {

    void EdgeIDPPrior::computeError() {
        const VertexPointInvDepth *vIDP = static_cast<const VertexPointInvDepth *>(_vertices[0]);
        _error(0) = vIDP->estimate() - _measurement;
    }

    /*
    void EdgeIDPPrior::linearizeOplus() {
        _jacobianOplusXi.setZero();
        _jacobianOplusXi(0) = 1;
    }
     */

    /**
     * Erorr = pi(Px)-obs
     */
    void EdgePRIDP::computeError() {
        const VertexPointInvDepth *vIDP = dynamic_cast<const VertexPointInvDepth *>( _vertices[0]);
        const VertexPR *vPR0 = dynamic_cast<const VertexPR *>(_vertices[1]);
        const VertexPR *vPRi = dynamic_cast<const VertexPR *>(_vertices[2]);

        // point inverse depth in reference KF
        double rho = vIDP->estimate();
        if (rho < 1e-6) {
            // LOG(WARNING) << "Inv depth should not be negative: " << rho << endl;
            return;
        }

        // point coordinate in reference KF, body
        Vec3 P0(x, y, 1.0);
        P0 = P0 * (1.0f / rho);
        Vec3 Pw = vPR0->estimate().inverse() * P0;
        Vec3 Pi = vPRi->estimate() * Pw;

        if (Pi[2] < 0) {
            // LOG(WARNING) << "projected depth should not be negative: " << Pi.transpose() << endl;
            return;
        }

        double xi = Pi[0] / Pi[2];
        double yi = Pi[1] / Pi[2];
        double u = cam->fxl() * xi + cam->cxl();
        double v = cam->fyl() * yi + cam->cyl();
        _error = Vec2(u, v) - _measurement;
    }

    void EdgeProjectPoseOnly::computeError() {
        const VertexPR *vPR = static_cast<VertexPR *> (_vertices[0]);
        SE3 Tcw = vPR->estimate();
        Vec3 pc = Tcw * pw;
        pc = pc * (1.0 / pc[2]);
        if (pc[2] < 0) {
            LOG(WARNING) << "invalid depth: " << pc[2] << endl;
            depthValid = false;
            return;
        }
        double u = fx * pc[0] + cx;
        double v = fy * pc[1] + cy;
        _error = Vec2(u, v) - _measurement;
    }

    void EdgeProjectPoseOnlySim3::computeError() {

        const VertexSim3 *vSim3 = static_cast<VertexSim3 *> (vertex(0));
        Sim3 Scw = vSim3->estimate();

        Vec3 pc = Scw.scale() * Scw.rotationMatrix() * pw + Scw.translation();
        pc = pc * (1.0 / pc[2]);

        if (pc[2] < 0) {
            LOG(WARNING) << "invalid depth: " << pc[2] << endl;
            setLevel(1);
            depthValid = false;
            return;
        }

        double u = fx * pc[0] + cx;
        double v = fy * pc[1] + cy;
        _error = Vec2(u, v) - _measurement;
    }
}
