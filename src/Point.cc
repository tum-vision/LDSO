#include "Feature.h"
#include "Point.h"
#include "Frame.h"
#include "internal/PointHessian.h"
#include "internal/GlobalCalib.h"

using namespace ldso::internal;
namespace ldso {
    unsigned long Point::mNextId = 0;

    Point::Point(shared_ptr<Feature> hostFeature) {
        mHostFeature = hostFeature;
        if (hostFeature->ip)    // create from immature point
            mpPH = shared_ptr<PointHessian>(new PointHessian(hostFeature->ip));
        else {
            LOG(ERROR) << "map point created without immature point, this should not happen!" << endl;
            mpPH = shared_ptr<PointHessian>(new PointHessian());
        }
        mpPH->point = hostFeature->point;
        id = mNextId++;
    }

    Point::Point() {
        id = mNextId++;
    }

    void Point::ReleasePH() {
        if (mpPH) {
            mpPH->point = nullptr;
            mpPH = nullptr;
        }
    }

    void Point::ComputeWorldPos() {
        shared_ptr<Feature> feat = mHostFeature.lock();
        if (feat) {
            shared_ptr<Frame> frame = feat->host.lock();
            if (!frame)
                return;
            Sim3 Twc = frame->getPoseOpti().inverse();
            Vec3 Kip = 1.0 / feat->invD * Vec3(
                    fxiG[0] * feat->uv[0] + cxiG[0],
                    fyiG[0] * feat->uv[1] + cyiG[0],
                    1);
            mWorldPos = Twc * Kip;
        }
    }

    void Point::save(ofstream &fout) {
        fout.write((char *) &id, sizeof(id));
        fout.write((char *) &status, sizeof(status));
    }

    void Point::load(ifstream &fin, vector<shared_ptr<Frame>> &allKFs) {

        fin.read((char *) &id, sizeof(id));
        fin.read((char *) &status, sizeof(status));
    }
}