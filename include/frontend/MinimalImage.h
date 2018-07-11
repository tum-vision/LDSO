#pragma once
#ifndef LDSO_MINIMAL_IMAGE_H_
#define LDSO_MINIMAL_IMAGE_H_

#include <algorithm>
#include "NumTypes.h"

using namespace std;

namespace ldso {

    template<typename T>
    class MinimalImage {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        int w;
        int h;
        T *data;

        /*
         * creates minimal image with own memory
         */
        inline MinimalImage(int w_, int h_) : w(w_), h(h_) {
            data = new T[w * h];
            ownData = true;
        }

        /*
         * creates minimal image wrapping around existing memory
         */
        inline MinimalImage(int w_, int h_, T *data_) : w(w_), h(h_) {
            data = data_;
            ownData = false;
        }

        inline ~MinimalImage() {
            if (ownData) delete[] data;
        }

        inline MinimalImage *getClone() {
            MinimalImage *clone = new MinimalImage(w, h);
            memcpy(clone->data, data, sizeof(T) * w * h);
            return clone;
        }


        inline T &at(int x, int y) { return data[(int) x + ((int) y) * w]; }

        inline T &at(int i) { return data[i]; }

        inline void setBlack() {
            memset(data, 0, sizeof(T) * w * h);
        }

        inline void setConst(T val) {
            for (int i = 0; i < w * h; i++) data[i] = val;
        }

        inline void setPixel1(const float &u, const float &v, T val) {
            at(u + 0.5f, v + 0.5f) = val;
        }

        inline void setPixel4(const float &u, const float &v, T val) {
            at(u + 1.0f, v + 1.0f) = val;
            at(u + 1.0f, v) = val;
            at(u, v + 1.0f) = val;
            at(u, v) = val;
        }

        inline void setPixel9(const int &u, const int &v, T val) {
            at(u + 1, v - 1) = val;
            at(u + 1, v) = val;
            at(u + 1, v + 1) = val;
            at(u, v - 1) = val;
            at(u, v) = val;
            at(u, v + 1) = val;
            at(u - 1, v - 1) = val;
            at(u - 1, v) = val;
            at(u - 1, v + 1) = val;
        }

        inline void setPixelCirc(const int &u, const int &v, T val) {
            for (int i = -3; i <= 3; i++) {
                at(u + 3, v + i) = val;
                at(u - 3, v + i) = val;
                at(u + 2, v + i) = val;
                at(u - 2, v + i) = val;

                at(u + i, v - 3) = val;
                at(u + i, v + 3) = val;
                at(u + i, v - 2) = val;
                at(u + i, v + 2) = val;
            }
        }

    private:
        bool ownData;
    };

    typedef Eigen::Matrix<unsigned char, 3, 1> Vec3b;
    typedef MinimalImage<float> MinimalImageF;
    typedef MinimalImage<Vec3f> MinimalImageF3;
    typedef MinimalImage<unsigned char> MinimalImageB;
    typedef MinimalImage<Vec3b> MinimalImageB3;
    typedef MinimalImage<unsigned short> MinimalImageB16;
}

#endif // LDSO_MINIMAL_IMAGE_H_
