#include "Tracker.h"
#include "Config.h"
#include <iostream>
#include <algorithm>

ObjectTracker::ObjectTracker()
{
    State          = ETrackState::Idle;
    LostFrameCount = 0;
    bBackupActive  = false;

    // History reduced 500→200
    // saves ~35 MB memory
    BGSub =
        cv::createBackgroundSubtractorMOG2(
            200, 16, false);

    Scales = { 0.8, 0.9, 1.0, 1.1, 1.2 };

    LoadFaceDetector();
}

void ObjectTracker::LoadFaceDetector()
{
    std::vector<std::string> Paths = {
        "C:/opencv/sources/data/haarcascades/"
        "haarcascade_frontalface_default.xml",
        "C:/opencv_build_new/install/etc/"
        "haarcascades/"
        "haarcascade_frontalface_default.xml",
        "haarcascade_frontalface_default.xml"
    };

    for (auto& P : Paths)
    {
        if (FaceDetector.load(P))
        {
            bFaceDetectorLoaded = true;
            std::cout
                << "Face detector loaded!\n";
            return;
        }
    }
    std::cout
        << "Face detector not found"
        << " - using motion only\n";
}

void ObjectTracker::ReinitTrackers(
    const cv::Mat& Frame,
    const cv::Rect& Box)
{
    PrimaryTracker =
        cv::TrackerCSRT::create();
    PrimaryTracker->init(Frame, Box);

    BackupTracker =
        cv::TrackerKCF::create();
    BackupTracker->init(Frame, Box);
    bBackupActive = true;

    std::cout << "CSRT+KCF initialized\n";
}

bool ObjectTracker::Init(
    const cv::Mat& Frame,
    const cv::Rect& BBox,
    ETrackerType Type)
{
    TrackerType = Type;

    cv::Rect SafeBox = BBox &
        cv::Rect(0, 0,
            Frame.cols, Frame.rows);
    if (SafeBox.area() <= 0)
        return false;

    // Save original template
    ObjTemplate = Frame(SafeBox).clone();

    // Save original size for
    // size constraint check
    OriginalWidth  = SafeBox.width;
    OriginalHeight = SafeBox.height;

    // Save 3D color histogram
    cv::Mat HSV;
    cv::cvtColor(Frame, HSV,
        cv::COLOR_BGR2HSV);

    cv::Mat ROI = HSV(SafeBox);
    int HistSize[] = { 8, 8, 8 };
    float HRange[] = { 0, 180 };
    float SRange[] = { 0, 256 };
    float VRange[] = { 0, 256 };
    const float* Ranges[] = {
        HRange, SRange, VRange };
    int Channels[] = { 0, 1, 2 };

    cv::calcHist(&ROI, 1, Channels,
        cv::Mat(), ObjHistogram,
        3, HistSize, Ranges);
    cv::normalize(ObjHistogram,
        ObjHistogram, 0, 1,
        cv::NORM_MINMAX);

    if (Type == ETrackerType::CSRT)
    {
        ReinitTrackers(Frame, SafeBox);
    }
    else if (Type == ETrackerType::KCF)
    {
        PrimaryTracker =
            cv::TrackerKCF::create();
        PrimaryTracker->init(
            Frame, SafeBox);
        bBackupActive = false;
        std::cout << "KCF initialized\n";
    }
    else if (Type == ETrackerType::CAMSHIFT)
    {
        cv::Mat HSV2;
        cv::cvtColor(Frame, HSV2,
            cv::COLOR_BGR2HSV);

        cv::Mat ROI2    = HSV2(SafeBox);
        int   HistSize2 = 16;
        float Range2[]  = { 0, 180 };
        const float* Ranges2 = Range2;
        int Channels2 = 0;

        cv::calcHist(&ROI2, 1, &Channels2,
            cv::Mat(), RoiHist,
            1, &HistSize2, &Ranges2);
        cv::normalize(RoiHist, RoiHist,
            0, 255, cv::NORM_MINMAX);

        TrackWindow   = SafeBox;
        bBackupActive = false;
        std::cout << "CamShift init\n";
    }

    CurrentBox     = SafeBox;
    State          = ETrackState::Tracking;
    LostFrameCount = 0;
    KFilter.Init(SafeBox);
    return true;
}

// ============================================
// Check if candidate box matches
// selected object by:
// 1. Color similarity
// 2. Size similarity
// ============================================
bool ObjectTracker::IsCorrectObject(
    const cv::Mat& Frame,
    const cv::Rect& Box)
{
    if (Box.area() < 100) return false;

    cv::Rect SafeBox = Box &
        cv::Rect(0, 0,
            Frame.cols, Frame.rows);
    if (SafeBox.area() < 100) return false;

    // Check 1 — Color match
    double ColorScore =
        CompareHistogram(Frame, SafeBox);
    if (ColorScore < 0.45)
    {
        std::cout
            << "Rejected: color mismatch "
            << (int)(ColorScore * 100)
            << "%\n";
        return false;
    }

    // Check 2 — Size match
    // Allow 50% size variation
    float WRatio =
        (float)SafeBox.width /
        OriginalWidth;
    float HRatio =
        (float)SafeBox.height /
        OriginalHeight;

    if (WRatio < 0.5f || WRatio > 2.0f ||
        HRatio < 0.5f || HRatio > 2.0f)
    {
        std::cout
            << "Rejected: size mismatch\n";
        return false;
    }

    return true;
}

cv::Rect ObjectTracker::AdaptScale(
    const cv::Mat& Frame,
    const cv::Rect& Box)
{
    if (ObjTemplate.empty()) return Box;

    double BestScore = -1;
    cv::Rect BestBox = Box;

    for (double Scale : Scales)
    {
        int NewW = (int)(Box.width  * Scale);
        int NewH = (int)(Box.height * Scale);
        int NewX = Box.x +
            (Box.width  - NewW) / 2;
        int NewY = Box.y +
            (Box.height - NewH) / 2;

        cv::Rect ScaledBox(
            NewX, NewY, NewW, NewH);
        ScaledBox &= cv::Rect(0, 0,
            Frame.cols, Frame.rows);

        if (ScaledBox.area() < 100)
            continue;

        double Score =
            CompareHistogram(
                Frame, ScaledBox);

        if (Score > BestScore)
        {
            BestScore = Score;
            BestBox   = ScaledBox;
        }
    }
    return BestBox;
}

bool ObjectTracker::UpdatePrimary(
    const cv::Mat& Frame,
    cv::Rect& OutBox)
{
    if (!PrimaryTracker) return false;

    bool bOK =
        PrimaryTracker->update(
            Frame, OutBox);
    if (!bOK) return false;

    OutBox &= cv::Rect(0, 0,
        Frame.cols, Frame.rows);
    if (OutBox.area() < 100) return false;

    // Validate it is correct object
    if (!IsCorrectObject(Frame, OutBox))
        return false;

    OutBox = AdaptScale(Frame, OutBox);
    return true;
}

bool ObjectTracker::UpdateBackup(
    const cv::Mat& Frame,
    cv::Rect& OutBox)
{
    if (!BackupTracker ||
        !bBackupActive) return false;

    bool bOK =
        BackupTracker->update(
            Frame, OutBox);
    if (!bOK) return false;

    OutBox &= cv::Rect(0, 0,
        Frame.cols, Frame.rows);
    if (OutBox.area() < 100) return false;

    // Validate it is correct object
    if (!IsCorrectObject(Frame, OutBox))
        return false;

    return true;
}

bool ObjectTracker::UpdateCamShift(
    const cv::Mat& Frame,
    cv::Rect& OutBox)
{
    if (RoiHist.empty()) return false;

    cv::Mat HSV, BackProj;
    cv::cvtColor(Frame, HSV,
        cv::COLOR_BGR2HSV);

    float Range[] = { 0, 180 };
    const float* Ranges = Range;
    int Channels = 0;

    cv::calcBackProject(&HSV, 1,
        &Channels, RoiHist,
        BackProj, &Ranges);

    cv::TermCriteria TC(
        cv::TermCriteria::EPS |
        cv::TermCriteria::COUNT,
        10, 1);

    cv::Rect SearchWin  = TrackWindow;
    SearchWin.x        -= 20;
    SearchWin.y        -= 20;
    SearchWin.width    += 40;
    SearchWin.height   += 40;
    SearchWin &= cv::Rect(0, 0,
        Frame.cols, Frame.rows);

    if (SearchWin.area() <= 0)
        return false;

    cv::Mat SearchBP = BackProj(SearchWin);
    cv::Rect LocalWin(
        0, 0,
        SearchWin.width,
        SearchWin.height);

    cv::RotatedRect RR =
        cv::CamShift(SearchBP,
            LocalWin, TC);

    OutBox    = RR.boundingRect();
    OutBox.x += SearchWin.x;
    OutBox.y += SearchWin.y;
    OutBox   &= cv::Rect(0, 0,
        Frame.cols, Frame.rows);

    if (OutBox.area() < 100) return false;
    TrackWindow = OutBox;

    cv::Mat Region = BackProj(
        OutBox & cv::Rect(0, 0,
            Frame.cols, Frame.rows));
    cv::Scalar Mean = cv::mean(Region);
    return Mean[0] > 30.0;
}

double ObjectTracker::CompareHistogram(
    const cv::Mat& Frame,
    const cv::Rect& Box)
{
    if (ObjHistogram.empty()) return 0.0;

    cv::Rect SafeBox = Box &
        cv::Rect(0, 0,
            Frame.cols, Frame.rows);
    if (SafeBox.area() <= 0) return 0.0;

    cv::Mat HSV;
    cv::cvtColor(Frame, HSV,
        cv::COLOR_BGR2HSV);

    cv::Mat ROI = HSV(SafeBox);
    int HistSize[] = { 8, 8, 8 };
    float HRange[] = { 0, 180 };
    float SRange[] = { 0, 256 };
    float VRange[] = { 0, 256 };
    const float* Ranges[] = {
        HRange, SRange, VRange };
    int Channels[] = { 0, 1, 2 };

    cv::Mat Hist;
    cv::calcHist(&ROI, 1, Channels,
        cv::Mat(), Hist,
        3, HistSize, Ranges);
    cv::normalize(Hist, Hist,
        0, 1, cv::NORM_MINMAX);

    return cv::compareHist(
        ObjHistogram, Hist,
        cv::HISTCMP_CORREL);
}

bool ObjectTracker::FindByFace(
    const cv::Mat& Frame,
    cv::Rect& OutBox)
{
    if (!bFaceDetectorLoaded) return false;

    cv::Mat Gray;
    cv::cvtColor(Frame, Gray,
        cv::COLOR_BGR2GRAY);
    cv::equalizeHist(Gray, Gray);

    std::vector<cv::Rect> Faces;
    FaceDetector.detectMultiScale(
        Gray, Faces,
        1.1, 3, 0,
        cv::Size(30, 30));

    if (Faces.empty()) return false;

    // Find face that best matches
    // selected object color AND size
    double BestScore = 0;
    cv::Rect BestFace;
    bool bFound = false;

    for (auto& Face : Faces)
    {
        // Check size matches original
        float WRatio =
            (float)Face.width /
            OriginalWidth;
        float HRatio =
            (float)Face.height /
            OriginalHeight;

        if (WRatio < 0.4f || WRatio > 2.5f ||
            HRatio < 0.4f || HRatio > 2.5f)
            continue;

        double ColorScore =
            CompareHistogram(Frame, Face);

        if (ColorScore > BestScore)
        {
            BestScore = ColorScore;
            BestFace  = Face;
            bFound    = true;
        }
    }

    if (!bFound) return false;

    OutBox = BestFace;
    std::cout
        << "Face detected! Score: "
        << (int)(BestScore * 100)
        << "%\n";
    return true;
}

bool ObjectTracker::FindByMotion(
    const cv::Mat& Frame,
    cv::Rect& OutBox)
{
    if (!BGSub) return false;

    cv::Mat FGMask;
    BGSub->apply(Frame, FGMask);

    cv::erode(FGMask, FGMask,
        cv::Mat(), cv::Point(-1,-1), 2);
    cv::dilate(FGMask, FGMask,
        cv::Mat(), cv::Point(-1,-1), 3);

    std::vector<std::vector<cv::Point>>
        Contours;
    cv::findContours(FGMask, Contours,
        cv::RETR_EXTERNAL,
        cv::CHAIN_APPROX_SIMPLE);

    if (Contours.empty()) return false;

    double BestScore = 0;
    cv::Rect BestBox;
    bool bFound = false;

    for (auto& C : Contours)
    {
        double Area = cv::contourArea(C);
        if (Area < 500) continue;

        cv::Rect MotionBox =
            cv::boundingRect(C);

        // Check size matches original
        float WRatio =
            (float)MotionBox.width /
            OriginalWidth;
        float HRatio =
            (float)MotionBox.height /
            OriginalHeight;

        if (WRatio < 0.3f || WRatio > 3.0f ||
            HRatio < 0.3f || HRatio > 3.0f)
            continue;

        // Check color matches original
        double ColorScore =
            CompareHistogram(
                Frame, MotionBox);

        if (ColorScore > BestScore)
        {
            BestScore = ColorScore;
            BestBox   = MotionBox;
            bFound    = true;
        }
    }

    // Need good color match
    if (!bFound || BestScore < 0.4)
        return false;

    OutBox = BestBox;
    std::cout
        << "Motion detected! Score: "
        << (int)(BestScore * 100)
        << "%\n";
    return true;
}

bool ObjectTracker::FindByTemplateAndColor(
    const cv::Mat& Frame,
    cv::Rect& OutBox)
{
    if (ObjTemplate.empty()) return false;

    if (Frame.cols < ObjTemplate.cols ||
        Frame.rows < ObjTemplate.rows)
        return false;

    cv::Mat Result;
    cv::matchTemplate(Frame,
        ObjTemplate, Result,
        cv::TM_CCOEFF_NORMED);

    double BestScore = 0;
    cv::Rect BestBox;

    for (int i = 0; i < 5; i++)
    {
        double MinVal, MaxVal;
        cv::Point MinLoc, MaxLoc;
        cv::minMaxLoc(Result,
            &MinVal, &MaxVal,
            &MinLoc, &MaxLoc);

        if (MaxVal < 0.5) break;

        cv::Rect CandBox(
            MaxLoc.x, MaxLoc.y,
            ObjTemplate.cols,
            ObjTemplate.rows);
        CandBox &= cv::Rect(0, 0,
            Frame.cols, Frame.rows);

        // Verify color AND size
        if (!IsCorrectObject(Frame, CandBox))
        {
            cv::rectangle(Result,
                cv::Point(
                    MaxLoc.x -
                    ObjTemplate.cols / 2,
                    MaxLoc.y -
                    ObjTemplate.rows / 2),
                cv::Point(
                    MaxLoc.x +
                    ObjTemplate.cols / 2,
                    MaxLoc.y +
                    ObjTemplate.rows / 2),
                cv::Scalar(0), -1);
            continue;
        }

        double ColorScore =
            CompareHistogram(
                Frame, CandBox);

        double CombinedScore =
            MaxVal * 0.5 +
            ColorScore * 0.5;

        if (CombinedScore > BestScore)
        {
            BestScore = CombinedScore;
            BestBox   = CandBox;
        }

        cv::rectangle(Result,
            cv::Point(
                MaxLoc.x -
                ObjTemplate.cols / 2,
                MaxLoc.y -
                ObjTemplate.rows / 2),
            cv::Point(
                MaxLoc.x +
                ObjTemplate.cols / 2,
                MaxLoc.y +
                ObjTemplate.rows / 2),
            cv::Scalar(0), -1);
    }

    if (BestScore < 0.55) return false;

    OutBox = BestBox;
    std::cout
        << "Template match! Score: "
        << (int)(BestScore * 100)
        << "%\n";
    return true;
}

bool ObjectTracker::TryReacquire(
    const cv::Mat& Frame)
{
    cv::Rect NewBox;

    // Method 1 — Face detection
    // (checks color+size match)
    if (FindByFace(Frame, NewBox))
    {
        ReinitTrackers(Frame, NewBox);
        CurrentBox     = NewBox;
        State          = ETrackState::Tracking;
        LostFrameCount = 0;
        KFilter.Init(NewBox);
        std::cout
            << "RECOVERED by face!\n";
        return true;
    }

    // Method 2 — Motion detection
    // (checks color+size match)
    if (FindByMotion(Frame, NewBox))
    {
        ReinitTrackers(Frame, NewBox);
        CurrentBox     = NewBox;
        State          = ETrackState::Tracking;
        LostFrameCount = 0;
        KFilter.Init(NewBox);
        std::cout
            << "RECOVERED by motion!\n";
        return true;
    }

    // Method 3 — Template + Color
    if (FindByTemplateAndColor(
        Frame, NewBox))
    {
        ReinitTrackers(Frame, NewBox);
        CurrentBox     = NewBox;
        State          = ETrackState::Tracking;
        LostFrameCount = 0;
        KFilter.Init(NewBox);
        std::cout
            << "RECOVERED by template!\n";
        return true;
    }

    std::cout << "Searching...\n";
    return false;
}

cv::Rect ObjectTracker::Update(
    const cv::Mat& Frame)
{
    if (State == ETrackState::Idle)
        return cv::Rect();

    if (BGSub &&
        State != ETrackState::Tracking)
    {
        cv::Mat FGMask;
        BGSub->apply(Frame, FGMask);
    }

    cv::Rect PrimaryBox, BackupBox;
    bool bPrimaryOK = false;
    bool bBackupOK  = false;

    if (TrackerType == ETrackerType::CSRT ||
        TrackerType == ETrackerType::KCF)
    {
        bPrimaryOK = UpdatePrimary(
            Frame, PrimaryBox);

        if (!bPrimaryOK && bBackupActive)
            bBackupOK = UpdateBackup(
                Frame, BackupBox);
    }
    else if (TrackerType ==
        ETrackerType::CAMSHIFT)
    {
        bPrimaryOK = UpdateCamShift(
            Frame, PrimaryBox);
    }

    cv::Rect TrackerBox;
    bool bOK = false;

    if (bPrimaryOK)
    {
        TrackerBox = PrimaryBox;
        bOK        = true;
    }
    else if (bBackupOK)
    {
        TrackerBox = BackupBox;
        bOK        = true;
        std::cout << "Backup used!\n";

        if (PrimaryTracker)
        {
            PrimaryTracker.release();
            PrimaryTracker =
                cv::TrackerCSRT::create();
            PrimaryTracker->init(
                Frame, BackupBox);
        }
    }

    if (bOK)
    {
        cv::Point LastC(
            CurrentBox.x +
            CurrentBox.width  / 2,
            CurrentBox.y +
            CurrentBox.height / 2);
        cv::Point NewC(
            TrackerBox.x +
            TrackerBox.width  / 2,
            TrackerBox.y +
            TrackerBox.height / 2);
        float Dist =
            (float)cv::norm(LastC - NewC);
        if (Dist > MAX_JUMP_DISTANCE)
            bOK = false;
    }

    if (bOK)
    {
        CurrentBox =
            KFilter.Update(TrackerBox);
        State          = ETrackState::Tracking;
        LostFrameCount = 0;
    }
    else
    {
        LostFrameCount++;

        if (LostFrameCount <= MAX_LOST_FRAMES)
        {
            CurrentBox = KFilter.Predict();
            State      = ETrackState::Occluded;
            std::cout << "Occluded ["
                << LostFrameCount << "]\n";
        }
        else
        {
            State = ETrackState::Lost;
            TryReacquire(Frame);
        }
    }
    return CurrentBox;
}

bool ObjectTracker::Reinit(
    const cv::Mat& Frame,
    const cv::Rect& NewBBox)
{
    return Init(Frame, NewBBox, TrackerType);
}

void ObjectTracker::Reset()
{
    KFilter.Reset();
    PrimaryTracker.release();
    BackupTracker.release();
    RoiHist.release();
    ObjTemplate.release();
    ObjHistogram.release();
    State          = ETrackState::Idle;
    LostFrameCount = 0;
    CurrentBox     = cv::Rect();
    bBackupActive  = false;
    OriginalWidth  = 0;
    OriginalHeight = 0;
}

void ObjectTracker::Draw(cv::Mat& Frame)
{
    if (State == ETrackState::Idle)
        return;

    cv::Scalar  Color;
    std::string Text;

    switch (State)
    {
    case ETrackState::Tracking:
        Color = cv::Scalar(0, 255, 0);
        Text  = "TRACKING";
        break;
    case ETrackState::Occluded:
        Color = cv::Scalar(0, 165, 255);
        Text  = "OCCLUDED";
        break;
    case ETrackState::Lost:
        Color = cv::Scalar(0, 0, 255);
        Text  = "SEARCHING...";
        break;
    default: return;
    }

    cv::rectangle(Frame,
        CurrentBox, Color, 2);

    if (SHOW_KALMAN_PRED &&
        State == ETrackState::Occluded)
    {
        cv::Rect P =
            KFilter.GetPredictedBox();
        cv::rectangle(Frame, P,
            cv::Scalar(255, 255, 0), 1);
        cv::putText(Frame, "KALMAN",
            cv::Point(P.x, P.y - 5),
            cv::FONT_HERSHEY_SIMPLEX,
            0.4,
            cv::Scalar(255, 255, 0), 1);
    }

    cv::Point C(
        CurrentBox.x + CurrentBox.width  / 2,
        CurrentBox.y + CurrentBox.height / 2);

    cv::line(Frame,
        cv::Point(C.x - 15, C.y),
        cv::Point(C.x + 15, C.y),
        Color, 2);
    cv::line(Frame,
        cv::Point(C.x, C.y - 15),
        cv::Point(C.x, C.y + 15),
        Color, 2);

    std::string TypeText;
    switch (TrackerType)
    {
    case ETrackerType::CSRT:
        TypeText = "[CSRT+KCF]";
        break;
    case ETrackerType::KCF:
        TypeText = "[KCF]";
        break;
    case ETrackerType::CAMSHIFT:
        TypeText = "[CAMSHIFT]";
        break;
    }

    cv::putText(Frame,
        Text + " " + TypeText,
        cv::Point(CurrentBox.x,
            CurrentBox.y - 10),
        cv::FONT_HERSHEY_SIMPLEX,
        0.6, Color, 2);
}
