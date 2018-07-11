#pragma once
#ifndef LDSO_SETTING_H_
#define LDSO_SETTING_H_

/// a lot many parameters set in DSO
namespace ldso {

    const int PYR_LEVELS = 6;  // total image pyramids, note not all are used during tracking
    const int NUM_THREADS = 6;

    // the config bits in solver
    const int SOLVER_SVD = 1;
    const int SOLVER_ORTHOGONALIZE_SYSTEM = 2;
    const int SOLVER_ORTHOGONALIZE_POINTMARG = 4;
    const int SOLVER_ORTHOGONALIZE_FULL = 8;
    const int SOLVER_SVD_CUT7 = 16;
    const int SOLVER_REMOVE_POSEPRIOR = 32;
    const int SOLVER_USE_GN = 64;
    const int SOLVER_FIX_LAMBDA = 128;
    const int SOLVER_ORTHOGONALIZE_X = 256;
    const int SOLVER_MOMENTUM = 512;
    const int SOLVER_STEPMOMENTUM = 1024;
    const int SOLVER_ORTHOGONALIZE_X_LATER = 2048;

    // constants to scale the parameters in optimization
    const float SCALE_IDEPTH = 1.0f;       // scales internal value to idepth.
    const float SCALE_XI_ROT = 1.0f;       //
    const float SCALE_XI_TRANS = 0.5f;     //
    const float SCALE_F = 50.0f;
    const float SCALE_C = 50.0f;
    const float SCALE_W = 1.0f;
    const float SCALE_A = 10.0f;           //
    const float SCALE_B = 1000.0f;         //

    // inverse version
    const float SCALE_IDEPTH_INVERSE = (1.0f / SCALE_IDEPTH);
    const float SCALE_XI_ROT_INVERSE = (1.0f / SCALE_XI_ROT);
    const float SCALE_XI_TRANS_INVERSE = (1.0f / SCALE_XI_TRANS);
    const float SCALE_F_INVERSE = (1.0f / SCALE_F);
    const float SCALE_C_INVERSE = (1.0f / SCALE_C);
    const float SCALE_W_INVERSE = (1.0f / SCALE_W);
    const float SCALE_A_INVERSE = (1.0f / SCALE_A);
    const float SCALE_B_INVERSE = (1.0f / SCALE_B);

    // the detail setting variables
    extern int pyrLevelsUsed;
    extern float setting_keyframesPerSecond;
    extern bool setting_realTimeMaxKF;
    extern float setting_maxShiftWeightT;
    extern float setting_maxShiftWeightR;
    extern float setting_maxShiftWeightRT;
    extern float setting_maxAffineWeight;
    extern float setting_kfGlobalWeight;
    extern float setting_idepthFixPrior;
    extern float setting_idepthFixPriorMargFac;
    extern float setting_initialRotPrior;
    extern float setting_initialTransPrior;
    extern float setting_initialAffBPrior;
    extern float setting_initialAffAPrior;
    extern float setting_initialCalibHessian;
    extern int setting_solverMode;
    extern double setting_solverModeDelta;
    extern float setting_minIdepthH_act;
    extern float setting_minIdepthH_marg;
    extern int setting_margPointVisWindow;
    extern float setting_maxIdepth;
    extern float setting_maxPixSearch;
    extern float setting_desiredImmatureDensity;                    // done
    extern float setting_desiredPointDensity;                       // done
    extern float setting_minPointsRemaining;
    extern float setting_maxLogAffFacInWindow;
    extern int setting_minFrames;
    extern int setting_maxFrames;
    extern int setting_minFrameAge;
    extern int setting_maxOptIterations;
    extern int setting_minOptIterations;
    extern float setting_thOptIterations;
    extern float setting_outlierTH;
    extern float setting_outlierTHSumComponent;
    extern float setting_outlierSmoothnessTH; // higher -> more strict
    extern int setting_killOverexposed;
    extern int setting_killOverexposedMode;
    extern int setting_pattern;
    extern float setting_margWeightFac;
    extern int setting_discreteSeachItsOnPointActivation;
    extern int setting_GNItsOnPointActivation;
    extern float setting_SmoothnessErrorPixelTH;
    extern float setting_SmoothnessEMinInlierPercentage;
    extern float setting_SmoothnessEGoodInlierPercentage;
    extern float setting_minTraceQuality;
    extern int setting_minTraceTestRadius;
    extern float setting_reTrackThreshold;
    extern int setting_minGoodActiveResForMarg;
    extern int setting_minGoodResForMarg;
    extern int setting_minInlierVotesForMarg;
    extern float setting_minRelBSForMarg;
    extern int setting_photometricCalibration;
    extern bool setting_useExposure;
    extern float setting_affineOptModeA;
    extern float setting_affineOptModeB;
    extern float setting_affineOptModeA_huberTH;
    extern float setting_affineOptModeB_huberTH;
    extern int setting_gammaWeightsPixelSelect;
    extern bool setting_relinAlways;
    extern bool setting_fixCalib;
    extern bool setting_activateAllOnMarg;
    extern bool setting_forceAceptStep;
    extern float setting_useDepthWeightsCoarse;
    extern bool setting_dilateDoubleCoarse;
    extern float setting_huberTH;
    extern bool setting_logStuff;   // whether to log
    extern float benchmarkSetting_fxfyfac;
    extern int benchmarkSetting_width;
    extern int benchmarkSetting_height;
    extern float benchmark_varNoise;
    extern float benchmark_varBlurNoise;
    extern int benchmark_noiseGridsize;
    extern float benchmark_initializerSlackFactor;
    extern float setting_frameEnergyTHConstWeight;
    extern float setting_frameEnergyTHN;
    extern float setting_frameEnergyTHFacMean;
    extern float setting_frameEnergyTHFacMedian;
    extern float setting_overallEnergyTHWeight;
    extern float setting_coarseCutoffTH;
    extern float setting_minGradHistCut;
    extern float setting_minGradHistAdd;
    extern float setting_fixGradTH;
    extern float setting_gradDownweightPerLevel;
    extern bool setting_selectDirectionDistribution;
    extern int setting_pixelSelectionUseFast;
    extern float setting_trace_stepsize;
    extern int setting_trace_GNIterations;
    extern float setting_trace_GNThreshold;
    extern float setting_trace_extraSlackOnTH;
    extern float setting_trace_slackInterval;
    extern float setting_trace_minImprovementFactor;
    extern bool setting_render_displayCoarseTrackingFull;
    extern bool setting_render_renderWindowFrames;
    extern bool setting_render_plotTrackingFull;
    extern bool setting_render_display3D;
    extern bool setting_render_displayResidual;
    extern bool setting_render_displayVideo;
    extern bool setting_render_displayDepth;
    extern bool setting_fullResetRequested;
    extern bool setting_debugout_runquiet;
    extern bool disableAllDisplay;
    extern bool disableReconfigure;
    extern bool setting_onlyLogKFPoses;
    extern bool debugSaveImages;
    extern int sparsityFactor;
    extern bool goStepByStep;
    extern bool plotStereoImages;
    extern bool multiThreading;
    extern float freeDebugParam1;
    extern float freeDebugParam2;
    extern float freeDebugParam3;
    extern float freeDebugParam4;
    extern float freeDebugParam5;
    extern int benchmarkSpecialOption;
    extern bool setting_pause;
    extern int setting_pointSelection;      // 0-DSO's selection. 1-LDSO's selection, 2-Random selection

    const int patternNum = 8;
    const int patternPadding = 2;

    // patterns, see Settings.cpp
    extern int staticPattern[10][40][2];
    extern int staticPatternNum[10];
    extern int staticPatternPadding[10];

    // loop closing setting
    // whether to use loop closing, if set to false, the program runs as original DSO
    // enabled by default
    extern bool setting_enableLoopClosing;

    // if fast loop closing is enabled, not all the keyframes in DSO will be corrected
    // and we will also have less constraints
    // NOTE if you don't enable fast loop closing, you would probably wait a little time for the pose graph optimization
    // loops will not be immediately closed after detection!
    extern bool setting_fastLoopClosing;

    // whether to show the loop closing, if true will show the loop keyframes in a opencv window
    // this is only for debugging (and for plotting when writing a paper)
    extern bool setting_showLoopClosing;

    // use the ninth pattern (described in DSO's paper)
#define patternP staticPattern[8]

}

#endif // LDSO_SETTING_H_
