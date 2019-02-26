#pragma once
#ifndef LDSO_DATASET_READER_H_
#define LDSO_DATASET_READER_H_

#include <sstream>
#include <fstream>
#include <dirent.h>
#include <algorithm>
#include <boost/format.hpp>

#include "Settings.h"
#include "frontend/Undistort.h"
#include "frontend/ImageRW.h"
#include "frontend/ImageAndExposure.h"

#include "internal/GlobalFuncs.h"
#include "internal/GlobalCalib.h"

#if HAS_ZIPLIB

#include "zip.h"

#endif

#include <iostream>

using namespace std;

using namespace ldso;
using namespace ldso::internal;

inline int getdir(std::string dir, std::vector<std::string> &files) {
    DIR *dp;
    struct dirent *dirp;
    if ((dp = opendir(dir.c_str())) == NULL) {
        return -1;
    }

    while ((dirp = readdir(dp)) != NULL) {
        std::string name = std::string(dirp->d_name);
        if (name != "." && name != ".." && name.substr(name.size() - 3, name.size()) == "jpg")
            files.push_back(name);
    }
    closedir(dp);
    std::sort(files.begin(), files.end());
    if (dir.at(dir.length() - 1) != '/') dir = dir + "/";
    for (unsigned int i = 0; i < files.size(); i++) {
        if (files[i].at(0) != '/')
            files[i] = dir + files[i];
    }

    LOG(INFO) << "files size: " << files.size() << endl;
    return files.size();
}


struct PrepImageItem {
    int id;
    bool isQueud;
    ImageAndExposure *pt;

    inline PrepImageItem(int _id) {
        id = _id;
        isQueud = false;
        pt = 0;
    }

    inline void release() {
        if (pt != 0) delete pt;
        pt = 0;
    }
};


class ImageFolderReader {

public:
    enum DatasetType {
        TUM_MONO,
        KITTI,
        EUROC
    };

    ImageFolderReader(DatasetType datasetType,
                      std::string path, std::string calibFile, std::string gammaFile,
                      std::string vignetteFile) {
        this->datasetType = datasetType;
        this->path = path;
        this->calibfile = calibFile;

#if HAS_ZIPLIB
        ziparchive = 0;
        databuffer = 0;
#endif

        isZipped = (path.length() > 4 && path.substr(path.length() - 4) == ".zip");

        if (datasetType == TUM_MONO) {
            if (isZipped) {
#if HAS_ZIPLIB
                int ziperror = 0;
                ziparchive = zip_open(path.c_str(), ZIP_RDONLY, &ziperror);
                if (ziperror != 0) {
                    printf("ERROR %d reading archive %s!\n", ziperror, path.c_str());
                    exit(1);
                }

                files.clear();
                int numEntries = zip_get_num_entries(ziparchive, 0);
                for (int k = 0; k < numEntries; k++) {
                    const char *name = zip_get_name(ziparchive, k, ZIP_FL_ENC_STRICT);
                    std::string nstr = std::string(name);
                    if (nstr == "." || nstr == "..") continue;
                    files.push_back(name);
                }

                printf("got %d entries and %d files!\n", numEntries, (int) files.size());
                std::sort(files.begin(), files.end());
#else
                LOG(FATAL)<<("ERROR: cannot read .zip archive, as compile without ziplib!\n");
#endif
            } else
                getdir(path, files);
        }

        undistort = Undistort::getUndistorterForFile(calibFile, gammaFile, vignetteFile);

        widthOrg = undistort->getOriginalSize()[0];
        heightOrg = undistort->getOriginalSize()[1];
        width = undistort->getSize()[0];
        height = undistort->getSize()[1];

        // load timestamps if possible.
        if (datasetType == TUM_MONO) {
            loadTimestamps();
        } else if (datasetType == EUROC) {
            loadTimestampsEUROC();
        } else if (datasetType == KITTI) {
            loadTimestampsKitti();
        }
    }

    ~ImageFolderReader() {
#if HAS_ZIPLIB
        if (ziparchive != 0) zip_close(ziparchive);
        if (databuffer != 0) delete databuffer;
#endif
        delete undistort;
    };

    Eigen::VectorXf getOriginalCalib() {
        return undistort->getOriginalParameter().cast<float>();
    }

    Eigen::Vector2i getOriginalDimensions() {
        return undistort->getOriginalSize();
    }

    void getCalibMono(Eigen::Matrix3f &K, int &w, int &h) {
        K = undistort->getK().cast<float>();
        w = undistort->getSize()[0];
        h = undistort->getSize()[1];
    }

    void setGlobalCalibration() {
        int w_out, h_out;
        Eigen::Matrix3f K;
        getCalibMono(K, w_out, h_out);
        setGlobalCalib(w_out, h_out, K);
    }

    int getNumImages() {
        return files.size();
    }

    double getTimestamp(int id) {
        if (timestamps.size() == 0) return id * 0.1f;
        if (id >= (int) timestamps.size()) return 0;
        if (id < 0) return 0;
        return timestamps[id];
    }


    void prepImage(int id, bool as8U = false) {

    }


    MinimalImageB *getImageRaw(int id) {
        return getImageRaw_internal(id, 0);
    }

    ImageAndExposure *getImage(int id, bool forceLoadDirectly = false) {
        return getImage_internal(id, 0);
    }


    inline float *getPhotometricGamma() {
        if (undistort == 0 || undistort->photometricUndist == 0) return 0;
        return undistort->photometricUndist->getG();
    }


    // undistorter. [0] always exists, [1-2] only when MT is enabled.
    Undistort *undistort;
private:


    MinimalImageB *getImageRaw_internal(int id, int unused) {
        if (!isZipped) {
            // CHANGE FOR ZIP FILE
            return IOWrap::readImageBW_8U(files[id]);
        } else {
#if HAS_ZIPLIB
            if (databuffer == 0) databuffer = new char[widthOrg * heightOrg * 6 + 10000];
            zip_file_t *fle = zip_fopen(ziparchive, files[id].c_str(), 0);
            long readbytes = zip_fread(fle, databuffer, (long) widthOrg * heightOrg * 6 + 10000);

            if (readbytes > (long) widthOrg * heightOrg * 6) {
                printf("read %ld/%ld bytes for file %s. increase buffer!!\n", readbytes,
                       (long) widthOrg * heightOrg * 6 + 10000, files[id].c_str());
                delete[] databuffer;
                databuffer = new char[(long) widthOrg * heightOrg * 30];
                fle = zip_fopen(ziparchive, files[id].c_str(), 0);
                readbytes = zip_fread(fle, databuffer, (long) widthOrg * heightOrg * 30 + 10000);

                if (readbytes > (long) widthOrg * heightOrg * 30) {
                    printf("buffer still to small (read %ld/%ld). abort.\n", readbytes,
                           (long) widthOrg * heightOrg * 30 + 10000);
                    exit(1);
                }
            }

            return IOWrap::readStreamBW_8U(databuffer, readbytes);
#else
            printf("ERROR: cannot read .zip archive, as compile without ziplib!\n");
            exit(1);
#endif
        }
    }


    ImageAndExposure *getImage_internal(int id, int unused) {
        MinimalImageB *minimg = getImageRaw_internal(id, 0);
        ImageAndExposure *ret2 = undistort->undistort<unsigned
        char>(
                minimg,
                (exposures.size() == 0 ? 1.0f : exposures[id]),
                (timestamps.size() == 0 ? 0.0 : timestamps[id]));
        delete minimg;
        return ret2;
    }

    inline void loadTimestampsEUROC() {

        LOG(INFO) << "loading EuRoC time stamps!" << endl;
        std::ifstream tr;
        std::string timesFile = path + "/data.csv";
        tr.open(timesFile.c_str());
        if (!tr) {
            LOG(INFO) << "cannot find timestamp file! Something maybe wrong in dataset setting!" << endl;
            LOG(INFO) << timesFile << endl;
            return;
        }

        while (!tr.eof() && tr.good()) {
            std::string line;
            char buf[1000];
            tr.getline(buf, 1000);

            double stamp;
            char filename[256];
            if (line[0] == '#')
                continue;

            if (2 == sscanf(buf, "%lf,%s", &stamp, filename)) {
                timestamps.push_back(stamp * 1e-9);
                files.push_back(path + "/data/" + string(filename));
            }
        }
        tr.close();
        LOG(INFO) << "Load total " << timestamps.size() << " data entries." << endl;
    }

    inline void loadTimestampsKitti() {
        LOG(INFO) << "Loading Kitti timestamps!" << endl;
        std::ifstream tr;
        std::string timesFile = path + "/times.txt";

        tr.open(timesFile.c_str());
        if (!tr) {
            LOG(INFO) << "cannot find timestamp file at " << path + "/times.txt" << endl;
            return;
        }

        while (!tr.eof() && tr.good()) {

            char buf[1000] = {0};
            tr.getline(buf, 1000);

            if (buf[0] == 0)
                break;

            double stamp = atof(buf);

            if (std::isnan(stamp))
                break;

            timestamps.push_back(stamp);
        }
        tr.close();

        // get the files
        boost::format fmt("%s/image_0/%06d.png");
        for (size_t i = 0; i < timestamps.size(); i++) {
            files.push_back((fmt % path % i).str());
        }

        LOG(INFO) << "Load total " << timestamps.size() << " data entries." << endl;
    }

    inline void loadTimestamps() {
        std::ifstream tr;
        std::string timesFile = path.substr(0, path.find_last_of('/')) + "/times.txt";
        tr.open(timesFile.c_str());

        if (!tr) {
            // try last directory
            timesFile = path.substr(0, path.find("images")) + "times.txt";
            LOG(INFO) << "trying " << timesFile << endl;
            tr.close();
            tr.open(timesFile.c_str());
        }

        while (!tr.eof() && tr.good()) {
            std::string line;
            char buf[1000];
            tr.getline(buf, 1000);

            int id;
            double stamp;
            float exposure = 0;

            if (3 == sscanf(buf, "%d %lf %f", &id, &stamp, &exposure)) {
                timestamps.push_back(stamp);
                exposures.push_back(exposure);
            } else if (2 == sscanf(buf, "%d %lf", &id, &stamp)) {
                timestamps.push_back(stamp);
                exposures.push_back(exposure);
            }
        }
        tr.close();

        LOG(INFO) << "files: " << files.size() << " timesstamps: " << timestamps.size() << ", exposures: "
                  << exposures.size() << endl;

        // check if exposures are correct, (possibly skip)
        bool exposuresGood = ((int) exposures.size() == (int) getNumImages());
        for (int i = 0; i < (int) exposures.size(); i++) {
            if (exposures[i] == 0) {
                // fix!
                float sum = 0, num = 0;
                if (i > 0 && exposures[i - 1] > 0) {
                    sum += exposures[i - 1];
                    num++;
                }
                if (i + 1 < (int) exposures.size() && exposures[i + 1] > 0) {
                    sum += exposures[i + 1];
                    num++;
                }

                if (num > 0)
                    exposures[i] = sum / num;
            }

            if (exposures[i] == 0) exposuresGood = false;
        }


        if ((int) getNumImages() != (int) timestamps.size()) {
            printf("set timestamps and exposures to zero!\n");
            exposures.clear();
            timestamps.clear();
        }

        if ((int) getNumImages() != (int) exposures.size() || !exposuresGood) {
            printf("set EXPOSURES to zero!\n");
            exposures.clear();
        }

        printf("got %d images and %d timestamps and %d exposures.!\n", (int) getNumImages(), (int) timestamps.size(),
               (int) exposures.size());
    }


    std::vector<ImageAndExposure *> preloadedImages;
    std::vector<std::string> files;
    std::vector<double> timestamps;
    std::vector<float> exposures;
    DatasetType datasetType;

    int width, height;
    int widthOrg, heightOrg;

    std::string path;
    std::string calibfile;

    bool isZipped;

#if HAS_ZIPLIB
    zip_t *ziparchive;
    char *databuffer;
#endif
};

#endif // LDSO_DATASET_READER_H_
