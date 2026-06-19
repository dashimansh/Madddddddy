#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <atomic>
#include <mutex>
#include "Config.h"
#include "Detector.h"
#include "Tracker.h"
#include "FrameStabilizer.h"
#include "ThreadedCamera.h"

// ============================================
// Mouse drawing state
// ============================================
struct MouseDrawData
{
    std::atomic<bool> bDrawing{ false };
    std::atomic<bool> bROIReady{ false };
    cv::Point         StartPoint{ 0, 0 };
    cv::Point         EndPoint{ 0, 0 };
    cv::Rect          DrawnROI;
    std::mutex        Mutex;
};

MouseDrawData GMouse;

void OnMouse(
    int Event, int X, int Y,
    int Flags, void* UserData)
{
    X = std::max(0, X);
    Y = std::max(0, Y);

    if (Event == cv::EVENT_LBUTTONDOWN)
    {
        std::lock_guard<std::mutex>
            Lock(GMouse.Mutex);
        GMouse.StartPoint = cv::Point(X,Y);
        GMouse.EndPoint   = cv::Point(X,Y);
        GMouse.bDrawing   = true;
        GMouse.bROIReady  = false;
    }
    else if (Event == cv::EVENT_MOUSEMOVE
        && GMouse.bDrawing.load())
    {
        std::lock_guard<std::mutex>
            Lock(GMouse.Mutex);
        GMouse.EndPoint = cv::Point(X,Y);
    }
    else if (Event == cv::EVENT_LBUTTONUP
        && GMouse.bDrawing.load())
    {
        std::lock_guard<std::mutex>
            Lock(GMouse.Mutex);
        GMouse.EndPoint  = cv::Point(X,Y);
        GMouse.bDrawing  = false;

        int RX = std::min(
            GMouse.StartPoint.x,
            GMouse.EndPoint.x);
        int RY = std::min(
            GMouse.StartPoint.y,
            GMouse.EndPoint.y);
        int RW = std::abs(
            GMouse.EndPoint.x -
            GMouse.StartPoint.x);
        int RH = std::abs(
            GMouse.EndPoint.y -
            GMouse.StartPoint.y);

        if (RW > 10 && RH > 10)
        {
            GMouse.DrawnROI =
                cv::Rect(RX,RY,RW,RH);
            GMouse.bROIReady = true;
            std::cout << "ROI: "
                << RW << "x"
                << RH << "\n";
        }
    }
}

// ============================================
// FPS counter
// ============================================
class FPSCounter
{
public:
    void Tick()
    {
        auto Now =
            std::chrono::steady_clock::now();
        double Elapsed =
            std::chrono::duration<double>(
                Now - Last).count();
        Last  = Now;
        Alpha = 0.1;
        FPS   = Alpha * (1.0 / Elapsed) +
                (1.0 - Alpha) * FPS;
    }
    double Get() const { return FPS; }

private:
    std::chrono::steady_clock::time_point
        Last =
        std::chrono::steady_clock::now();
    double FPS   = 0.0;
    double Alpha = 0.1;
};

// ============================================
// Draw live drag rectangle
// ============================================
void DrawLiveROI(cv::Mat& Frame)
{
    if (!GMouse.bDrawing.load()) return;
    std::lock_guard<std::mutex>
        Lock(GMouse.Mutex);

    cv::Point P1 = GMouse.StartPoint;
    cv::Point P2 = GMouse.EndPoint;

    cv::rectangle(Frame, P1, P2,
        cv::Scalar(0,255,255), 2);

    int CL = 10;
    cv::line(Frame, P1,
        {P1.x+CL, P1.y},
        cv::Scalar(0,255,0), 3);
    cv::line(Frame, P1,
        {P1.x, P1.y+CL},
        cv::Scalar(0,255,0), 3);
    cv::line(Frame, P2,
        {P2.x-CL, P2.y},
        cv::Scalar(0,255,0), 3);
    cv::line(Frame, P2,
        {P2.x, P2.y-CL},
        cv::Scalar(0,255,0), 3);

    int W = std::abs(P2.x-P1.x);
    int H = std::abs(P2.y-P1.y);
    cv::putText(Frame,
        std::to_string(W)+"x"+
        std::to_string(H),
        {std::min(P1.x,P2.x),
         std::min(P1.y,P2.y)-5},
        cv::FONT_HERSHEY_SIMPLEX,
        0.5, cv::Scalar(0,255,255), 1);
}

// ============================================
// Draw HUD — FPS + State + Controls
// ============================================
void DrawHUD(
    cv::Mat& Frame,
    double   FPS,
    ETrackState State,
    int Lost)
{
    cv::rectangle(Frame,
        cv::Rect(0,0,Frame.cols,70),
        cv::Scalar(0,0,0), -1);

    // FPS — color coded
    std::string FPSStr =
        "FPS: " + std::to_string((int)FPS);
    cv::Scalar FPSColor;
    if      (FPS >= 20)
        FPSColor = cv::Scalar(0,255,0);
    else if (FPS >= 10)
        FPSColor = cv::Scalar(0,165,255);
    else
        FPSColor = cv::Scalar(0,0,255);

    cv::putText(Frame, FPSStr,
        cv::Point(10,25),
        cv::FONT_HERSHEY_SIMPLEX,
        0.7, FPSColor, 2);

    // State
    std::string S;
    cv::Scalar  SC;
    switch (State)
    {
    case ETrackState::Idle:
        S  = "DRAG TO SELECT";
        SC = cv::Scalar(0,255,255);
        break;
    case ETrackState::Tracking:
        S  = "TRACKING";
        SC = cv::Scalar(0,255,0);
        break;
    case ETrackState::Occluded:
        S  = "OCCLUDED [" +
            std::to_string(Lost) +
            "/" +
            std::to_string(MAX_LOST_FRAMES)
            + "]";
        SC = cv::Scalar(0,165,255);
        break;
    case ETrackState::Lost:
        S  = "SEARCHING...";
        SC = cv::Scalar(0,0,255);
        break;
    }
    cv::putText(Frame,
        "STATE: " + S,
        cv::Point(160,25),
        cv::FONT_HERSHEY_SIMPLEX,
        0.55, SC, 2);

    cv::putText(Frame,
        "DRAG=Select  R=Reset  "
        "1=CSRT 2=KCF  Q=Quit",
        cv::Point(10,52),
        cv::FONT_HERSHEY_SIMPLEX,
        0.42,
        cv::Scalar(180,180,180), 1);
}

// ============================================
// MAIN
// ============================================
int main()
{
    std::cout
        << "Object Tracker\n"
        << "==============\n";

    // ================================================
    // SELECT MODE:
    // 1 = Webcam (laptop camera)
    // 2 = Video file
    // 3 = IP camera (RTSP)
    // ================================================
    const int MODE = 1;

    // Webcam index (0 = built-in)
    const int CAMERA_INDEX = 0;

    // Video file path
    const std::string VIDEO_PATH =
        "C:/Users/Victus/Downloads/video.mp4";

    // IP camera RTSP URL
    const std::string IP_URL =
        "rtsp://admin:admin123"
        "@192.168.1.64:554/stream";
    // ================================================

    ThreadedCamera  Camera;
    Detector        Det;
    ObjectTracker   Tracker;
    FrameStabilizer Stab;
    FPSCounter      FPS;

    // ============================================
    // Open camera based on MODE
    // ============================================
    if (MODE == 1)
    {
        std::cout
            << "Opening webcam...\n";
        if (!Camera.Open(CAMERA_INDEX))
        {
            std::cerr
                << "Failed to open webcam!\n"
                << "Check camera index\n";
            return -1;
        }
        std::cout << "Webcam opened!\n";
    }
    else if (MODE == 2)
    {
        std::cout
            << "Opening video: "
            << VIDEO_PATH << "\n";
        if (!Camera.OpenFile(VIDEO_PATH))
        {
            std::cerr
                << "Failed! Check path:\n"
                << VIDEO_PATH << "\n";
            return -1;
        }
        std::cout << "Video opened!\n";
    }
    else if (MODE == 3)
    {
        std::cout
            << "Connecting to IP camera:\n"
            << IP_URL << "\n";
        if (!Camera.OpenFile(IP_URL))
        {
            std::cerr
                << "Failed to connect!\n"
                << "Check IP and URL\n";
            return -1;
        }
        std::cout << "IP camera connected!\n";
    }
    else
    {
        std::cerr
            << "Invalid MODE!"
            << " Use 1, 2 or 3\n";
        return -1;
    }

    std::cout
        << "\nDrag mouse on object "
        << "to start tracking!\n"
        << "R=Reset  1=CSRT  "
        << "2=KCF  Q=Quit\n\n";

    cv::Mat Frame;
    int  FrameCount = 0;
    bool bStabilize = false;

    // Create display window
    cv::namedWindow(
        WINDOW_NAME,
        cv::WINDOW_NORMAL);
    cv::resizeWindow(
        WINDOW_NAME, 960, 540);

    // Register mouse callback
    cv::setMouseCallback(
        WINDOW_NAME,
        OnMouse, nullptr);

    // ============================================
    // MAIN LOOP
    // ============================================
    while (true)
    {
        if (!Camera.GetLatestFrame(Frame))
            continue;

        FrameCount++;
        FPS.Tick();

        // Use Frame directly when not
        // stabilizing — saves ~15 MB
        cv::Mat PF;
        if (bStabilize)
            PF = Stab.Stabilize(Frame);
        else
            PF = Frame;

        // ----------------------------------------
        // Check if ROI was drawn
        // ----------------------------------------
        if (GMouse.bROIReady.load())
        {
            cv::Rect DrawnROI;
            {
                std::lock_guard<std::mutex>
                    Lock(GMouse.Mutex);
                DrawnROI = GMouse.DrawnROI;
                GMouse.bROIReady = false;
            }

            DrawnROI &= cv::Rect(0,0,
                PF.cols, PF.rows);

            if (DrawnROI.area() > 100)
            {
                Tracker.Init(PF,
                    DrawnROI,
                    ETrackerType::CSRT);
                std::cout
                    << "Tracking started!\n";
            }
        }

        // ----------------------------------------
        // Frame skipping for performance
        // CSRT runs every FRAME_SKIP frames
        // Kalman fills skipped frames
        // ----------------------------------------
        bool bRunTracking =
            (FrameCount % FRAME_SKIP == 0);

        if (Tracker.GetState() !=
            ETrackState::Idle)
        {
            if (bRunTracking)
                Tracker.Update(PF);
            else
                Tracker.PredictOnly();
        }

        // ----------------------------------------
        // Draw everything
        // ----------------------------------------
        Tracker.Draw(PF);
        DrawLiveROI(PF);
        DrawHUD(PF,
            FPS.Get(),
            Tracker.GetState(),
            Tracker.GetLostCount());

        // Show instruction when idle
        if (Tracker.GetState() ==
            ETrackState::Idle &&
            !GMouse.bDrawing.load())
        {
            cv::putText(PF,
                "Drag mouse on object",
                {PF.cols/2 - 150,
                 PF.rows/2},
                cv::FONT_HERSHEY_SIMPLEX,
                0.8,
                cv::Scalar(0,255,255), 2);
        }

        cv::imshow(WINDOW_NAME, PF);
        char Key = (char)cv::waitKey(1);

        // ----------------------------------------
        // Keyboard controls
        // ----------------------------------------
        if (Key == 'r' || Key == 'R')
        {
            Tracker.Reset();
            Stab.Reset();
            std::cout << "Reset!\n";
        }
        else if (Key == 't' || Key == 'T')
        {
            bStabilize = !bStabilize;
            std::cout << "Stabilize: "
                << (bStabilize ?
                    "ON" : "OFF") << "\n";
        }
        else if (Key == '1')
        {
            if (Tracker.GetState() !=
                ETrackState::Idle)
            {
                Tracker.Init(PF,
                    Tracker.GetCurrentBox(),
                    ETrackerType::CSRT);
                std::cout << "CSRT\n";
            }
        }
        else if (Key == '2')
        {
            if (Tracker.GetState() !=
                ETrackState::Idle)
            {
                Tracker.Init(PF,
                    Tracker.GetCurrentBox(),
                    ETrackerType::KCF);
                std::cout << "KCF\n";
            }
        }
        else if (Key == '3')
        {
            if (Tracker.GetState() !=
                ETrackState::Idle)
            {
                Tracker.Init(PF,
                    Tracker.GetCurrentBox(),
                    ETrackerType::CAMSHIFT);
                std::cout << "CamShift\n";
            }
        }
        else if (Key == 'q' ||
                 Key == 'Q' ||
                 Key == 27)
            break;
    }

    Camera.Stop();
    cv::destroyAllWindows();
    return 0;
}
