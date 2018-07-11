#pragma once
#ifndef LDSO_IMAGE_AND_EXPOSURE_H_
#define LDSO_IMAGE_AND_EXPOSURE_H_

#include "NumTypes.h"

namespace ldso {

    class ImageAndExposure {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        float *image = nullptr;            // irradiance. between 0 and 256
        int w = 0, h = 0;                // width and height;
        double timestamp = 0;
        float exposure_time = 0;    // exposure time in ms.

        inline ImageAndExposure(int w_, int h_, double timestamp_ = 0) : w(w_), h(h_), timestamp(timestamp_) {
            image = new float[w * h];
            exposure_time = 1;
        }

        inline ~ImageAndExposure() {
            delete[] image;
        }

        inline void copyMetaTo(ImageAndExposure &other) {
            other.exposure_time = exposure_time;
        }

        inline ImageAndExposure *getDeepCopy() {
            ImageAndExposure *img = new ImageAndExposure(w, h, timestamp);
            img->exposure_time = exposure_time;
            memcpy(img->image, image, w * h * sizeof(float));
            return img;
        }
    };
}

#endif //  LDSO_IMAGE_AND_EXPOSURE_H_
