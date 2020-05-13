#include <thread>
#include <clocale>
#include <csignal>
#include <cstdlib>
#include <cstdio>
#include <unistd.h>
#include <sys/time.h>

#include <glog/logging.h>

#include "frontend/FullSystem.h"
#include "DatasetReader.h"

/*********************************************************************************
 * This program demonstrates how to run LDSO in TUM-Mono dataset
 * which is the default dataset in DSO and quite difficult because of low texture
 * Please specify the dataset directory below or by command line parameters
 *********************************************************************************/

using namespace std;
using namespace ldso;

std::string vignette = "/media/gaoxiang/Data1/Dataset/TUM-MONO/sequence_31/vignette.png";
std::string gammaCalib = "/media/gaoxiang/Data1/Dataset/TUM-MONO/sequence_31/pcalib.txt";
std::string source = "/media/gaoxiang/Data1/Dataset/TUM-MONO/sequence_31/";
std::string calib = "/media/gaoxiang/Data1/Dataset/TUM-MONO/sequence_31/camera.txt";
std::string output_file = "./results.txt";
std::string vocPath = "./vocab/orbvoc.dbow3";

double rescale = 1;
bool reversePlay = false;
bool disableROS = false;
int startIdx = 0;
int endIdx = 100000;
bool prefetch = false;
float playbackSpeed = 0;    // 0 for linearize (play as fast as possible, while sequentializing tracking & mapping). otherwise, factor on timestamps.
bool preload = false;
bool useSampleOutput = false;

using namespace ldso;

void settingsDefault(int preset) {
    printf("\n=============== PRESET Settings: ===============\n");
    if (preset == 0 || preset == 1) {
        printf("DEFAULT settings:\n"
               "- %s real-time enforcing\n"
               "- 2000 active points\n"
               "- 5-7 active frames\n"
               "- 1-6 LM iteration each KF\n"
               "- original image resolution\n", preset == 0 ? "no " : "1x");

        playbackSpeed = (preset == 0 ? 0 : 1);
        preload = preset == 1;
        setting_desiredImmatureDensity = 1500;
        setting_desiredPointDensity = 2000;
        setting_minFrames = 5;
        setting_maxFrames = 7;
        setting_maxOptIterations = 6;
        setting_minOptIterations = 1;

        setting_logStuff = false;
    }

    if (preset == 2 || preset == 3) {
        printf("FAST settings:\n"
               "- %s real-time enforcing\n"
               "- 800 active points\n"
               "- 4-6 active frames\n"
               "- 1-4 LM iteration each KF\n"
               "- 424 x 320 image resolution\n",
               preset == 2 ? "no " : "5x");

        playbackSpeed = (preset == 2 ? 0 : 5);
        preload = preset == 3;
        setting_desiredImmatureDensity = 600;
        setting_desiredPointDensity = 800;
        setting_minFrames = 4;
        setting_maxFrames = 6;
        setting_maxOptIterations = 4;
        setting_minOptIterations = 1;

        benchmarkSetting_width = 424;
        benchmarkSetting_height = 320;

        setting_logStuff = false;
    }

    printf("==============================================\n");
}

void parseArgument(char *arg) {
    int option;
    float foption;
    char buf[1000];


    if (1 == sscanf(arg, "sampleoutput=%d", &option)) {
        if (option == 1) {
            useSampleOutput = true;
            printf("USING SAMPLE OUTPUT WRAPPER!\n");
        }
        return;
    }

    if (1 == sscanf(arg, "quiet=%d", &option)) {
        if (option == 1) {
            setting_debugout_runquiet = true;
            printf("QUIET MODE, I'll shut up!\n");
        }
        return;
    }

    if (1 == sscanf(arg, "preset=%d", &option)) {
        settingsDefault(option);
        return;
    }


    if (1 == sscanf(arg, "rec=%d", &option)) {
        if (option == 0) {
            disableReconfigure = true;
            printf("DISABLE RECONFIGURE!\n");
        }
        return;
    }


    if (1 == sscanf(arg, "noros=%d", &option)) {
        if (option == 1) {
            disableROS = true;
            disableReconfigure = true;
            printf("DISABLE ROS (AND RECONFIGURE)!\n");
        }
        return;
    }

    if (1 == sscanf(arg, "nolog=%d", &option)) {
        if (option == 1) {
            setting_logStuff = false;
            printf("DISABLE LOGGING!\n");
        }
        return;
    }
    if (1 == sscanf(arg, "reversePlay=%d", &option)) {
        if (option == 1) {
            reversePlay = true;
            printf("REVERSE!\n");
        }
        return;
    }
    if (1 == sscanf(arg, "nogui=%d", &option)) {
        if (option == 1) {
            disableAllDisplay = true;
            printf("NO GUI!\n");
        }
        return;
    }
    if (1 == sscanf(arg, "nomt=%d", &option)) {
        if (option == 1) {
            multiThreading = false;
            printf("NO MultiThreading!\n");
        }
        return;
    }
    if (1 == sscanf(arg, "prefetch=%d", &option)) {
        if (option == 1) {
            prefetch = true;
            printf("PREFETCH!\n");
        }
        return;
    }
    if (1 == sscanf(arg, "start=%d", &option)) {
        startIdx = option;
        printf("START AT %d!\n", startIdx);
        return;
    }
    if (1 == sscanf(arg, "end=%d", &option)) {
        endIdx = option;
        printf("END AT %d!\n", endIdx);
        return;
    }
    if (1 == sscanf(arg, "loopclosing=%d", &option)) {
        if (option == 1) {
            setting_enableLoopClosing = true;
        } else {
            setting_enableLoopClosing = false;
        }
        printf("Loopclosing %s!\n", setting_enableLoopClosing ? "enabled" : "disabled");
        return;
    }

    if (1 == sscanf(arg, "files=%s", buf)) {
        source = buf;
        printf("loading data from %s!\n", source.c_str());
        return;
    }

    if (1 == sscanf(arg, "vocab=%s", buf)) {
        vocPath = buf;
        printf("loading vocabulary from %s!\n", vocPath.c_str());
        return;
    }

    if (1 == sscanf(arg, "calib=%s", buf)) {
        calib = buf;
        printf("loading calibration from %s!\n", calib.c_str());
        return;
    }

    if (1 == sscanf(arg, "vignette=%s", buf)) {
        vignette = buf;
        printf("loading vignette from %s!\n", vignette.c_str());
        return;
    }

    if (1 == sscanf(arg, "gamma=%s", buf)) {
        gammaCalib = buf;
        printf("loading gammaCalib from %s!\n", gammaCalib.c_str());
        return;
    }

    if (1 == sscanf(arg, "rescale=%f", &foption)) {
        rescale = foption;
        printf("RESCALE %f!\n", rescale);
        return;
    }

    if (1 == sscanf(arg, "speed=%f", &foption)) {
        playbackSpeed = foption;
        printf("PLAYBACK SPEED %f!\n", playbackSpeed);
        return;
    }

    if (1 == sscanf(arg, "output=%s", buf)) {
        output_file = buf;
        LOG(INFO) << "output set to " << output_file << endl;
        return;
    }

    if (1 == sscanf(arg, "save=%d", &option)) {
        if (option == 1) {
            debugSaveImages = true;
            if (42 == system("rm -rf images_out"))
                printf("system call returned 42 - what are the odds?. This is only here to shut up the compiler.\n");
            if (42 == system("mkdir images_out"))
                printf("system call returned 42 - what are the odds?. This is only here to shut up the compiler.\n");
            if (42 == system("rm -rf images_out"))
                printf("system call returned 42 - what are the odds?. This is only here to shut up the compiler.\n");
            if (42 == system("mkdir images_out"))
                printf("system call returned 42 - what are the odds?. This is only here to shut up the compiler.\n");
            printf("SAVE IMAGES!\n");
        }
        return;
    }

    if (1 == sscanf(arg, "mode=%d", &option)) {
        if (option == 0) {
            printf("PHOTOMETRIC MODE WITH CALIBRATION!\n");
        }
        if (option == 1) {
            printf("PHOTOMETRIC MODE WITHOUT CALIBRATION!\n");
            setting_photometricCalibration = 0;
            setting_affineOptModeA = 0; //-1: fix. >=0: optimize (with prior, if > 0).
            setting_affineOptModeB = 0; //-1: fix. >=0: optimize (with prior, if > 0).
        }
        if (option == 2) {
            printf("PHOTOMETRIC MODE WITH PERFECT IMAGES!\n");
            setting_photometricCalibration = 0;
            setting_affineOptModeA = -1; //-1: fix. >=0: optimize (with prior, if > 0).
            setting_affineOptModeB = -1; //-1: fix. >=0: optimize (with prior, if > 0).
            setting_minGradHistAdd = 3;
        }
        return;
    }

    printf("could not parse argument \"%s\"!!!!\n", arg);
}

int main(int argc, char **argv) {

    FLAGS_colorlogtostderr = true;
    for (int i = 1; i < argc; i++)
        parseArgument(argv[i]);

    // check setting conflicts
    if (setting_enableLoopClosing && (setting_pointSelection != 1)) {
        LOG(ERROR) << "Loop closing is enabled but point selection strategy is not set to LDSO, "
                      "use setting_pointSelection=1! please!" << endl;
        exit(-1);
    }

    if (setting_showLoopClosing) {
        LOG(WARNING) << "show loop closing results. The program will be paused when any loop is found" << endl;
    }

    shared_ptr<ImageFolderReader> reader(
            new ImageFolderReader(ImageFolderReader::TUM_MONO, source, calib, gammaCalib, vignette));

    reader->setGlobalCalibration();

    if (setting_photometricCalibration > 0 && reader->getPhotometricGamma() == 0) {
        LOG(ERROR) << "ERROR: dont't have photometric calibation. Need to use commandline options mode=1 or mode=2 ";
        exit(1);
    }

    int lstart = startIdx;
    int lend = endIdx;
    int linc = 1;
    if (reversePlay) {
        LOG(INFO) << "REVERSE!!!!";
        lstart = endIdx - 1;
        if (lstart >= reader->getNumImages())
            lstart = reader->getNumImages() - 1;
        lend = startIdx;
        linc = -1;
    }

    shared_ptr<ORBVocabulary> voc(new ORBVocabulary());
    voc->load(vocPath);

    shared_ptr<FullSystem> fullSystem(new FullSystem(voc));
    fullSystem->setGammaFunction(reader->getPhotometricGamma());
    fullSystem->linearizeOperation = (playbackSpeed == 0);

    shared_ptr<PangolinDSOViewer> viewer = nullptr;
    if (!disableAllDisplay) {
        viewer = shared_ptr<PangolinDSOViewer>(new PangolinDSOViewer(wG[0], hG[0], false));
        fullSystem->setViewer(viewer);
    } else {
        LOG(INFO) << "visualization is disabled!" << endl;
    }

    // to make MacOS happy: run this in dedicated thread -- and use this one to run the GUI.
    std::thread runthread([&]() {
        std::vector<int> idsToPlay;
        std::vector<double> timesToPlayAt;
        for (int i = lstart; i >= 0 && i < reader->getNumImages() && linc * i < linc * lend; i += linc) {
            idsToPlay.push_back(i);
            if (timesToPlayAt.size() == 0) {
                timesToPlayAt.push_back((double) 0);
            } else {
                double tsThis = reader->getTimestamp(idsToPlay[idsToPlay.size() - 1]);
                double tsPrev = reader->getTimestamp(idsToPlay[idsToPlay.size() - 2]);
                timesToPlayAt.push_back(timesToPlayAt.back() + fabs(tsThis - tsPrev) / playbackSpeed);
            }
        }


        std::vector<ImageAndExposure *> preloadedImages;
        if (preload) {
            printf("LOADING ALL IMAGES!\n");
            for (int ii = 0; ii < (int) idsToPlay.size(); ii++) {
                int i = idsToPlay[ii];
                preloadedImages.push_back(reader->getImage(i));
            }
        }

        struct timeval tv_start;
        gettimeofday(&tv_start, NULL);
        clock_t started = clock();
        double sInitializerOffset = 0;

        for (int ii = 0; ii < (int) idsToPlay.size(); ii++) {

            while (setting_pause == true) {
                usleep(5000);
            }

            if (!fullSystem->initialized)    // if not initialized: reset start time.
            {
                gettimeofday(&tv_start, NULL);
                started = clock();
                sInitializerOffset = timesToPlayAt[ii];
            }

            int i = idsToPlay[ii];

            ImageAndExposure *img;
            if (preload)
                img = preloadedImages[ii];
            else
                img = reader->getImage(i);

            bool skipFrame = false;
            if (playbackSpeed != 0) {
                struct timeval tv_now;
                gettimeofday(&tv_now, NULL);
                double sSinceStart = sInitializerOffset + ((tv_now.tv_sec - tv_start.tv_sec) +
                                                           (tv_now.tv_usec - tv_start.tv_usec) / (1000.0f * 1000.0f));

                if (sSinceStart < timesToPlayAt[ii]) {
                    // usleep((int) ((timesToPlayAt[ii] - sSinceStart) * 1000 * 1000));
                }
                else if (sSinceStart > timesToPlayAt[ii] + 0.5 + 0.1 * (ii % 2)) {
                    printf("SKIPFRAME %d (play at %f, now it is %f)!\n", ii, timesToPlayAt[ii], sSinceStart);
                    skipFrame = true;
                }
            }
            if (!skipFrame) {
                fullSystem->addActiveFrame(img, i);
            }
            delete img;

            if (fullSystem->initFailed || setting_fullResetRequested) {
                if (ii < 250 || setting_fullResetRequested) {
                    LOG(INFO) << "Init failed, RESETTING!";
                    fullSystem = shared_ptr<FullSystem>(new FullSystem(voc));
                    fullSystem->setGammaFunction(reader->getPhotometricGamma());
                    fullSystem->linearizeOperation = (playbackSpeed == 0);
                    if (viewer) {
                        viewer->reset();
                        sleep(1);
                        fullSystem->setViewer(viewer);
                    }
                    setting_fullResetRequested = false;
                }
            }

            if (fullSystem->isLost) {
                LOG(INFO) << "Lost!";
                break;
            }

        }

        fullSystem->blockUntilMappingIsFinished();

        clock_t ended = clock();
        struct timeval tv_end;
        gettimeofday(&tv_end, NULL);

        fullSystem->printResult(output_file, true);
        fullSystem->printResult(output_file + ".noloop", false);

        int numFramesProcessed = abs(idsToPlay[0] - idsToPlay.back());
        double numSecondsProcessed = fabs(reader->getTimestamp(idsToPlay[0]) - reader->getTimestamp(idsToPlay.back()));
        double MilliSecondsTakenSingle = 1000.0f * (ended - started) / (float) (CLOCKS_PER_SEC);
        double MilliSecondsTakenMT = sInitializerOffset + ((tv_end.tv_sec - tv_start.tv_sec) * 1000.0f +
                                                           (tv_end.tv_usec - tv_start.tv_usec) / 1000.0f);
        printf("\n======================"
               "\n%d Frames (%.1f fps)"
               "\n%.2fms per frame (single core); "
               "\n%.2fms per frame (multi core); "
               "\n%.3fx (single core); "
               "\n%.3fx (multi core); "
               "\n======================\n\n",
               numFramesProcessed, numFramesProcessed / numSecondsProcessed,
               MilliSecondsTakenSingle / numFramesProcessed,
               MilliSecondsTakenMT / (float) numFramesProcessed,
               1000 / (MilliSecondsTakenSingle / numSecondsProcessed),
               1000 / (MilliSecondsTakenMT / numSecondsProcessed));
        if (setting_logStuff) {
            std::ofstream tmlog;
            tmlog.open("logs/time.txt", std::ios::trunc | std::ios::out);
            tmlog << 1000.0f * (ended - started) / (float) (CLOCKS_PER_SEC * reader->getNumImages()) << " "
                  << ((tv_end.tv_sec - tv_start.tv_sec) * 1000.0f + (tv_end.tv_usec - tv_start.tv_usec) / 1000.0f) /
                     (float) reader->getNumImages() << "\n";
            tmlog.flush();
            tmlog.close();
        }
    });

    if (viewer)
        viewer->run();  // mac os should keep this in main thread.

    runthread.join();

    viewer->saveAsPLYFile("./pointcloud.ply");
    LOG(INFO) << "EXIT NOW!";
    return 0;
}
