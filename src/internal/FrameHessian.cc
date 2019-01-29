#include "internal/FrameHessian.h"
#include "internal/GlobalCalib.h"

#include <iostream>
#include <opencv2/opencv.hpp>


namespace ldso {

    namespace internal {

        void FrameHessian::setStateZero(const Vec10 &state_zero) {

            assert(state_zero.head<6>().squaredNorm() < 1e-20);

            this->state_zero = state_zero;

            for (int i = 0; i < 6; i++) {
                Vec6 eps;
                eps.setZero();
                eps[i] = 1e-3;
                SE3 EepsP = SE3::exp(eps);
                SE3 EepsM = SE3::exp(-eps);
                SE3 w2c_leftEps_P_x0 = (get_worldToCam_evalPT() * EepsP) * get_worldToCam_evalPT().inverse();
                SE3 w2c_leftEps_M_x0 = (get_worldToCam_evalPT() * EepsM) * get_worldToCam_evalPT().inverse();
                nullspaces_pose.col(i) = (w2c_leftEps_P_x0.log() - w2c_leftEps_M_x0.log()) / (2e-3);
            }

            // scale change
            SE3 w2c_leftEps_P_x0 = (get_worldToCam_evalPT());
            w2c_leftEps_P_x0.translation() *= 1.00001;
            w2c_leftEps_P_x0 = w2c_leftEps_P_x0 * get_worldToCam_evalPT().inverse();
            SE3 w2c_leftEps_M_x0 = (get_worldToCam_evalPT());
            w2c_leftEps_M_x0.translation() /= 1.00001;
            w2c_leftEps_M_x0 = w2c_leftEps_M_x0 * get_worldToCam_evalPT().inverse();
            nullspaces_scale = (w2c_leftEps_P_x0.log() - w2c_leftEps_M_x0.log()) / (2e-3);

            nullspaces_affine.setZero();
            nullspaces_affine.topLeftCorner<2, 1>() = Vec2(1, 0);
            assert(ab_exposure > 0);
            nullspaces_affine.topRightCorner<2, 1>() = Vec2(0, expf(aff_g2l_0().a) * ab_exposure);
        }

        void FrameHessian::makeImages(float *color, const shared_ptr<CalibHessian> &HCalib) {

            for (int i = 0; i < pyrLevelsUsed; i++) {
                dIp[i] = new Eigen::Vector3f[wG[i] * hG[i]];
                absSquaredGrad[i] = new float[wG[i] * hG[i]];
                memset(absSquaredGrad[i], 0, wG[i] * hG[i]);
                memset(dIp[i], 0, 3 * wG[i] * hG[i]);
            }
            dI = dIp[0];

            // make d0
            int w = wG[0];
            int h = hG[0];
            for (int i = 0; i < w * h; i++) {
                dI[i][0] = color[i];
            }

            for (int lvl = 0; lvl < pyrLevelsUsed; lvl++) {
                int wl = wG[lvl], hl = hG[lvl];
                Eigen::Vector3f *dI_l = dIp[lvl];

                float *dabs_l = absSquaredGrad[lvl];
                if (lvl > 0) {
                    int lvlm1 = lvl - 1;
                    int wlm1 = wG[lvlm1];
                    Eigen::Vector3f *dI_lm = dIp[lvlm1];


                    for (int y = 0; y < hl; y++)
                        for (int x = 0; x < wl; x++) {
                            dI_l[x + y * wl][0] = 0.25f * (dI_lm[2 * x + 2 * y * wlm1][0] +
                                                           dI_lm[2 * x + 1 + 2 * y * wlm1][0] +
                                                           dI_lm[2 * x + 2 * y * wlm1 + wlm1][0] +
                                                           dI_lm[2 * x + 1 + 2 * y * wlm1 + wlm1][0]);
                        }
                }

                for (int idx = wl; idx < wl * (hl - 1); idx++) {
                    float dx = 0.5f * (dI_l[idx + 1][0] - dI_l[idx - 1][0]);
                    float dy = 0.5f * (dI_l[idx + wl][0] - dI_l[idx - wl][0]);

                    if (std::isnan(dx) || std::fabs(dx) > 255.0) dx = 0;
                    if (std::isnan(dy) || std::fabs(dy) > 255.0) dy = 0;

                    dI_l[idx][1] = dx;
                    dI_l[idx][2] = dy;

                    dabs_l[idx] = dx * dx + dy * dy;

                    if (setting_gammaWeightsPixelSelect == 1 && HCalib != 0) {
                        float gw = HCalib->getBGradOnly((float) (dI_l[idx][0]));
                        dabs_l[idx] *=
                                gw * gw;    // convert to gradient of original color space (before removing response).
                        // if (std::isnan(dabs_l[idx])) dabs_l[idx] = 0;
                    }
                }
            }

            // === debug stuffs === //
            if (setting_enableLoopClosing && setting_showLoopClosing) {
                frame->imgDisplay = cv::Mat(hG[0], wG[0], CV_8UC3);
                uchar *data = frame->imgDisplay.data;
                for (int i = 0; i < w * h; i++) {
                    for (int c = 0; c < 3; c++) {
                        *data = color[i] > 255 ? 255 : uchar(color[i]);
                        data++;
                    }
                }
            }
        }

        void FrameHessian::takeData() {
            prior = getPrior().head<8>();
            delta = get_state_minus_stateZero().head<8>();
            delta_prior = (get_state() - getPriorZero()).head<8>();
        }
    }

}
