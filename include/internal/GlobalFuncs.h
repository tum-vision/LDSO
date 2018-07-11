#pragma once
#ifndef LDSO_GLOBAL_FUNCS_H_
#define LDSO_GLOBAL_FUNCS_H_

#include "NumTypes.h"

namespace ldso {

    // the global functions used in dso, for interpolations and visualizations
    namespace internal {

        // remove an element from a vector
        template<typename T>
        inline void deleteOut(std::vector<T *> &v, const int i) {
            v[i] = v.back();
            v.pop_back();
        }

        template<typename T>
        inline void deleteOut(std::vector<shared_ptr<T>> &v, const int i) {
            v[i] = v.back();
            v.pop_back();
        }

        template<typename T>
        inline void deleteOut(std::vector<shared_ptr<T>> &v, const shared_ptr<T> &e) {
            for (shared_ptr<T> &t: v) {
                if (t == e) {
                    t = v.back();
                    v.pop_back();
                    break;
                }
            }
        }

        // delete an element from a vector and keep its order
        template<typename T>
        inline void deleteOutOrder(std::vector<T> &v, const T element) {
            int i = -1;
            for (unsigned int k = 0; k < v.size(); k++) {
                if (v[k] == element) {
                    i = k;
                    break;
                }
            }
            assert(i != -1);

            for (unsigned int k = i + 1; k < v.size(); k++)
                v[k - 1] = v[k];
            v.pop_back();
        }

        // reads interpolated element from a float array
        EIGEN_ALWAYS_INLINE float
        getInterpolatedElement(const float *const mat, const float x, const float y, const int width) {
            int ix = (int) x;
            int iy = (int) y;
            float dx = x - ix;
            float dy = y - iy;
            float dxdy = dx * dy;
            const float *bp = mat + ix + iy * width;


            float res = dxdy * bp[1 + width]
                        + (dy - dxdy) * bp[width]
                        + (dx - dxdy) * bp[1]
                        + (1 - dx - dy + dxdy) * bp[0];
            return res;
        }

        // get interpolated value from Vector4f mat
        // return interpolated Vector3f by throwing away the last dimension
        EIGEN_ALWAYS_INLINE Eigen::Vector3f
        getInterpolatedElement43(const Eigen::Vector4f *const mat, const float x, const float y, const int width) {
            int ix = (int) x;
            int iy = (int) y;
            float dx = x - ix;
            float dy = y - iy;
            float dxdy = dx * dy;
            const Eigen::Vector4f *bp = mat + ix + iy * width;

            return dxdy * *(const Eigen::Vector3f *) (bp + 1 + width)
                   + (dy - dxdy) * *(const Eigen::Vector3f *) (bp + width)
                   + (dx - dxdy) * *(const Eigen::Vector3f *) (bp + 1)
                   + (1 - dx - dy + dxdy) * *(const Eigen::Vector3f *) (bp);
        }

        // get interpolated value from Vector3f mat
        EIGEN_ALWAYS_INLINE Eigen::Vector3f
        getInterpolatedElement33(const Eigen::Vector3f *const mat, const float x, const float y, const int width) {
            int ix = (int) x;
            int iy = (int) y;
            float dx = x - ix;
            float dy = y - iy;
            float dxdy = dx * dy;
            const Eigen::Vector3f *bp = mat + ix + iy * width;


            return dxdy * *(const Eigen::Vector3f *) (bp + 1 + width)
                   + (dy - dxdy) * *(const Eigen::Vector3f *) (bp + width)
                   + (dx - dxdy) * *(const Eigen::Vector3f *) (bp + 1)
                   + (1 - dx - dy + dxdy) * *(const Eigen::Vector3f *) (bp);
        }

        // interpolation with "and" mask
        EIGEN_ALWAYS_INLINE Eigen::Vector3f
        getInterpolatedElement33OverAnd(const Eigen::Vector3f *const mat, const bool *overMat, const float x,
                                        const float y, const int width, bool &over_out) {
            int ix = (int) x;
            int iy = (int) y;
            float dx = x - ix;
            float dy = y - iy;
            float dxdy = dx * dy;
            const Eigen::Vector3f *bp = mat + ix + iy * width;

            const bool *bbp = overMat + ix + iy * width;
            over_out = bbp[1 + width] && bbp[1] && bbp[width] && bbp[0];

            return dxdy * *(const Eigen::Vector3f *) (bp + 1 + width)
                   + (dy - dxdy) * *(const Eigen::Vector3f *) (bp + width)
                   + (dx - dxdy) * *(const Eigen::Vector3f *) (bp + 1)
                   + (1 - dx - dy + dxdy) * *(const Eigen::Vector3f *) (bp);
        }

        // interpolation with "or" mask
        EIGEN_ALWAYS_INLINE Eigen::Vector3f
        getInterpolatedElement33OverOr(const Eigen::Vector3f *const mat, const bool *overMat, const float x,
                                       const float y, const int width, bool &over_out) {
            int ix = (int) x;
            int iy = (int) y;
            float dx = x - ix;
            float dy = y - iy;
            float dxdy = dx * dy;
            const Eigen::Vector3f *bp = mat + ix + iy * width;

            const bool *bbp = overMat + ix + iy * width;
            over_out = bbp[1 + width] || bbp[1] || bbp[width] || bbp[0];

            return dxdy * *(const Eigen::Vector3f *) (bp + 1 + width)
                   + (dy - dxdy) * *(const Eigen::Vector3f *) (bp + width)
                   + (dx - dxdy) * *(const Eigen::Vector3f *) (bp + 1)
                   + (1 - dx - dy + dxdy) * *(const Eigen::Vector3f *) (bp);
        }

        EIGEN_ALWAYS_INLINE float
        getInterpolatedElement31(const Eigen::Vector3f *const mat, const float x, const float y, const int width) {
            int ix = (int) x;
            int iy = (int) y;
            float dx = x - ix;
            float dy = y - iy;
            float dxdy = dx * dy;
            const Eigen::Vector3f *bp = mat + ix + iy * width;


            return dxdy * (*(const Eigen::Vector3f *) (bp + 1 + width))[0]
                   + (dy - dxdy) * (*(const Eigen::Vector3f *) (bp + width))[0]
                   + (dx - dxdy) * (*(const Eigen::Vector3f *) (bp + 1))[0]
                   + (1 - dx - dy + dxdy) * (*(const Eigen::Vector3f *) (bp))[0];
        }

        EIGEN_ALWAYS_INLINE Eigen::Vector3f
        getInterpolatedElement13BiLin(const float *const mat, const float x, const float y, const int width) {
            int ix = (int) x;
            int iy = (int) y;
            const float *bp = mat + ix + iy * width;

            float tl = *(bp);
            float tr = *(bp + 1);
            float bl = *(bp + width);
            float br = *(bp + width + 1);

            float dx = x - ix;
            float dy = y - iy;
            float topInt = dx * tr + (1 - dx) * tl;
            float botInt = dx * br + (1 - dx) * bl;
            float leftInt = dy * bl + (1 - dy) * tl;
            float rightInt = dy * br + (1 - dy) * tr;

            return Eigen::Vector3f(
                    dx * rightInt + (1 - dx) * leftInt,
                    rightInt - leftInt,
                    botInt - topInt);
        }

        EIGEN_ALWAYS_INLINE Eigen::Vector3f
        getInterpolatedElement33BiLin(const Eigen::Vector3f *const mat, const float x, const float y, const int width) {
            int ix = (int) x;
            int iy = (int) y;
            const Eigen::Vector3f *bp = mat + ix + iy * width;

            float tl = (*(bp))[0];
            float tr = (*(bp + 1))[0];
            float bl = (*(bp + width))[0];
            float br = (*(bp + width + 1))[0];

            float dx = x - ix;
            float dy = y - iy;
            float topInt = dx * tr + (1 - dx) * tl;
            float botInt = dx * br + (1 - dx) * bl;
            float leftInt = dy * bl + (1 - dy) * tl;
            float rightInt = dy * br + (1 - dy) * tr;

            return Eigen::Vector3f(
                    dx * rightInt + (1 - dx) * leftInt,
                    rightInt - leftInt,
                    botInt - topInt);
        }

        EIGEN_ALWAYS_INLINE float
        getInterpolatedElement11Cub(const float *const p, const float x)   // for x=0, this returns p[1].
        {
            return p[1] + 0.5f * x * (p[2] - p[0] + x * (2.0f * p[0] - 5.0f * p[1] + 4.0f * p[2] - p[3] +
                                                         x * (3.0f * (p[1] - p[2]) + p[3] - p[0])));
        }

        EIGEN_ALWAYS_INLINE Eigen::Vector2f
        getInterpolatedElement12Cub(const float *const p, const float x) // for x=0, this returns p[1].
        {
            float c1 = 0.5f * (p[2] - p[0]);
            float c2 = p[0] - 2.5f * p[1] + 2 * p[2] - 0.5f * p[3];
            float c3 = 0.5f * (3.0f * (p[1] - p[2]) + p[3] - p[0]);
            float xx = x * x;
            float xxx = xx * x;
            return Eigen::Vector2f(p[1] + x * c1 + xx * c2 + xxx * c3, c1 + x * 2.0f * c2 + xx * 3.0f * c3);
        }

        EIGEN_ALWAYS_INLINE Eigen::Vector2f
        getInterpolatedElement32Cub(const Eigen::Vector3f *const p, const float x) // for x=0, this returns p[1].
        {
            float c1 = 0.5f * (p[2][0] - p[0][0]);
            float c2 = p[0][0] - 2.5f * p[1][0] + 2 * p[2][0] - 0.5f * p[3][0];
            float c3 = 0.5f * (3.0f * (p[1][0] - p[2][0]) + p[3][0] - p[0][0]);
            float xx = x * x;
            float xxx = xx * x;
            return Eigen::Vector2f(p[1][0] + x * c1 + xx * c2 + xxx * c3, c1 + x * 2.0f * c2 + xx * 3.0f * c3);
        }

        EIGEN_ALWAYS_INLINE float
        getInterpolatedElement11BiCub(const float *const mat, const float x, const float y, const int width) {
            int ix = (int) x;
            int iy = (int) y;
            float dx = x - ix;
            const float *bp = mat + ix + iy * width;

            float val[4];
            val[0] = getInterpolatedElement11Cub(bp - width - 1, dx);
            val[1] = getInterpolatedElement11Cub(bp - 1, dx);
            val[2] = getInterpolatedElement11Cub(bp + width - 1, dx);
            val[3] = getInterpolatedElement11Cub(bp + 2 * width - 1, dx);

            float dy = y - iy;
            return getInterpolatedElement11Cub(val, dy);
        }

        EIGEN_ALWAYS_INLINE Eigen::Vector3f
        getInterpolatedElement13BiCub(const float *const mat, const float x, const float y, const int width) {
            int ix = (int) x;
            int iy = (int) y;
            float dx = x - ix;
            const float *bp = mat + ix + iy * width;

            float val[4];
            float grad[4];
            Eigen::Vector2f v = getInterpolatedElement12Cub(bp - width - 1, dx);
            val[0] = v[0];
            grad[0] = v[1];

            v = getInterpolatedElement12Cub(bp - 1, dx);
            val[1] = v[0];
            grad[1] = v[1];

            v = getInterpolatedElement12Cub(bp + width - 1, dx);
            val[2] = v[0];
            grad[2] = v[1];

            v = getInterpolatedElement12Cub(bp + 2 * width - 1, dx);
            val[3] = v[0];
            grad[3] = v[1];

            float dy = y - iy;
            v = getInterpolatedElement12Cub(val, dy);

            return Eigen::Vector3f(v[0], getInterpolatedElement11Cub(grad, dy), v[1]);
        }

        EIGEN_ALWAYS_INLINE Eigen::Vector3f
        getInterpolatedElement33BiCub(const Eigen::Vector3f *const mat, const float x, const float y, const int width) {
            int ix = (int) x;
            int iy = (int) y;
            float dx = x - ix;
            const Eigen::Vector3f *bp = mat + ix + iy * width;

            float val[4];
            float grad[4];
            Eigen::Vector2f v = getInterpolatedElement32Cub(bp - width - 1, dx);
            val[0] = v[0];
            grad[0] = v[1];

            v = getInterpolatedElement32Cub(bp - 1, dx);
            val[1] = v[0];
            grad[1] = v[1];

            v = getInterpolatedElement32Cub(bp + width - 1, dx);
            val[2] = v[0];
            grad[2] = v[1];

            v = getInterpolatedElement32Cub(bp + 2 * width - 1, dx);
            val[3] = v[0];
            grad[3] = v[1];

            float dy = y - iy;
            v = getInterpolatedElement12Cub(val, dy);

            return Eigen::Vector3f(v[0], getInterpolatedElement11Cub(grad, dy), v[1]);
        }

        EIGEN_ALWAYS_INLINE Eigen::Vector4f
        getInterpolatedElement44(const Eigen::Vector4f *const mat, const float x, const float y, const int width) {
            int ix = (int) x;
            int iy = (int) y;
            float dx = x - ix;
            float dy = y - iy;
            float dxdy = dx * dy;
            const Eigen::Vector4f *bp = mat + ix + iy * width;


            return dxdy * *(bp + 1 + width)
                   + (dy - dxdy) * *(bp + width)
                   + (dx - dxdy) * *(bp + 1)
                   + (1 - dx - dy + dxdy) * *(bp);
        }

        EIGEN_ALWAYS_INLINE Eigen::Vector2f
        getInterpolatedElement42(const Eigen::Vector4f *const mat, const float x, const float y, const int width) {
            int ix = (int) x;
            int iy = (int) y;
            float dx = x - ix;
            float dy = y - iy;
            float dxdy = dx * dy;
            const Eigen::Vector4f *bp = mat + ix + iy * width;


            return dxdy * *(const Eigen::Vector2f *) (bp + 1 + width)
                   + (dy - dxdy) * *(const Eigen::Vector2f *) (bp + width)
                   + (dx - dxdy) * *(const Eigen::Vector2f *) (bp + 1)
                   + (1 - dx - dy + dxdy) * *(const Eigen::Vector2f *) (bp);
        }

        // -----------------------------------------------------------------
        // visualization functions
        // generate colors for visualization
        inline Vec3f makeRainbowf3F(float id) {
            id *= freeDebugParam3;
            if (id < 0) {
                return Vec3f(1, 1, 1);
            }

            int icP = id;
            float ifP = id - icP;
            icP = icP % 3;

            if (icP == 0) {
                return Vec3f((1 - ifP), ifP, 0);
            }
            if (icP == 1) {
                return Vec3f(0, (1 - ifP), ifP);
            }
            if (icP == 2) {
                return Vec3f(ifP, 0, (1 - ifP));
            }
            assert (false);
            return Vec3f(1, 1, 1);
        }

        inline Vec3b makeRainbow3B(float id) {
            id *= freeDebugParam3;
            if (!(id > 0)) {
                return Vec3b(255, 255, 255);
            }

            int icP = id;
            float ifP = id - icP;
            icP = icP % 3;

            if (icP == 0) {
                return Vec3b(255 * (1 - ifP), 255 * ifP, 0);
            }
            if (icP == 1) {
                return Vec3b(0, 255 * (1 - ifP), 255 * ifP);
            }
            if (icP == 2) {
                return Vec3b(255 * ifP, 0, 255 * (1 - ifP));
            }
            return Vec3b(255, 255, 255);
        }

        inline Vec3b makeJet3B(float id) {
            if (id <= 0) {
                return Vec3b(128, 0, 0);
            }
            if (id >= 1) {
                return Vec3b(0, 0, 128);
            }

            int icP = (id * 8);
            float ifP = (id * 8) - icP;

            if (icP == 0) {
                return Vec3b(255 * (0.5 + 0.5 * ifP), 0, 0);
            }
            if (icP == 1) {
                return Vec3b(255, 255 * (0.5 * ifP), 0);
            }
            if (icP == 2) {
                return Vec3b(255, 255 * (0.5 + 0.5 * ifP), 0);
            }
            if (icP == 3) {
                return Vec3b(255 * (1 - 0.5 * ifP), 255, 255 * (0.5 * ifP));
            }
            if (icP == 4) {
                return Vec3b(255 * (0.5 - 0.5 * ifP), 255, 255 * (0.5 + 0.5 * ifP));
            }
            if (icP == 5) {
                return Vec3b(0, 255 * (1 - 0.5 * ifP), 255);
            }
            if (icP == 6) {
                return Vec3b(0, 255 * (0.5 - 0.5 * ifP), 255);
            }
            if (icP == 7) {
                return Vec3b(0, 0, 255 * (1 - 0.5 * ifP));
            }
            return Vec3b(255, 255, 255);
        }

        inline Vec3b makeRedGreen3B(float val) // 0 = red, 1=green, 0.5=yellow.
        {
            if (val < 0) {
                return Vec3b(0, 0, 255);
            } else if (val < 0.5) {
                return Vec3b(0, 255 * 2 * val, 255);
            } else if (val < 1) {
                return Vec3b(0, 255, 255 - 255 * 2 * (val - 0.5));
            } else {
                return Vec3b(0, 255, 0);
            }

        }

        inline bool eigenTestNan(const MatXX &m, std::string msg) {
            bool foundNan = false;
            for (int y = 0; y < m.rows(); y++)
                for (int x = 0; x < m.cols(); x++) {
                    if (!std::isfinite((double) m(y, x))) {
                        foundNan = true;
                    }
                }
            if (foundNan) {
                printf("NAN in %s:\n", msg.c_str());
            }
            return foundNan;
        }
    }

}

#endif // LDSO_GLOBAL_FUNCS_H_
