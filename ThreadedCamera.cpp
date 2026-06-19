#include "ThreadedCamera.h"
#include "Config.h"
#include <iostream>
#include <thread>
#include <chrono>

ThreadedCamera::ThreadedCamera() {}

ThreadedCamera::~ThreadedCamera()
{
    Stop();
}

bool ThreadedCamera::Open(int ID)
{
    Cap.open(ID, cv::CAP_DSHOW);
    if (!Cap.isOpened())
    {
        Cap.open(ID);
        if (!Cap.isOpened())
        {
            std::cerr
                << "Cannot open camera\n";
            return false;
        }
    }
    SetupCamera();
    return true;
}

bool ThreadedCamera::OpenFile(
    const std::string& Path)
{
    Cap.open(Path);
    if (!Cap.isOpened()) return false;

    // ============================================
    // KEY FIX 1 — Minimize RTSP buffer
    // Buffer=1 means always get latest frame
    // not frames that are 2-3 seconds old!
    // ============================================
    Cap.set(cv::CAP_PROP_BUFFERSIZE, 1);

    Width  = (int)Cap.get(
        cv::CAP_PROP_FRAME_WIDTH);
    Height = (int)Cap.get(
        cv::CAP_PROP_FRAME_HEIGHT);
    FPS    = (int)Cap.get(
        cv::CAP_PROP_FPS);

    if (FPS <= 0 || FPS > 120)
        FPS = 25;

    std::cout << "IP Camera: "
        << Width << "x" << Height
        << " @ " << FPS << "fps\n";

    bRunning  = true;
    CamThread = std::thread(
        &ThreadedCamera::CaptureLoop,
        this);
    return true;
}

void ThreadedCamera::SetupCamera()
{
    Cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
    Cap.set(cv::CAP_PROP_FRAME_WIDTH,
        CAMERA_WIDTH);
    Cap.set(cv::CAP_PROP_FRAME_HEIGHT,
        CAMERA_HEIGHT);
    Cap.set(cv::CAP_PROP_FPS,
        CAMERA_FPS);

    Width  = (int)Cap.get(
        cv::CAP_PROP_FRAME_WIDTH);
    Height = (int)Cap.get(
        cv::CAP_PROP_FRAME_HEIGHT);
    FPS    = (int)Cap.get(
        cv::CAP_PROP_FPS);

    if (FPS <= 0 || FPS > 120)
        FPS = 30;

    std::cout << "Camera: "
        << Width << "x" << Height
        << " @ " << FPS << "fps\n";

    bRunning  = true;
    CamThread = std::thread(
        &ThreadedCamera::CaptureLoop,
        this);
}

void ThreadedCamera::CaptureLoop()
{
    while (bRunning)
    {
        cv::Mat Frame;

        // ============================================
        // KEY FIX 2 — Grab then retrieve
        // grab() discards frame from buffer fast
        // retrieve() only decodes the latest one
        // This skips stale frames efficiently!
        // ============================================
        if (!Cap.grab())
        {
            std::this_thread::sleep_for(
                std::chrono::milliseconds(5));
            continue;
        }

        // Only decode every frame
        // grab() already discarded old ones
        Cap.retrieve(Frame);

        if (Frame.empty())
        {
            std::this_thread::sleep_for(
                std::chrono::milliseconds(5));
            continue;
        }

        // ============================================
        // KEY FIX 3 — Resize to tracking resolution
        // IP cameras often send 1080p or 720p
        // We only need 640x360 for tracking!
        // This alone can double your FPS
        // ============================================
        if (Frame.cols > CAMERA_WIDTH ||
            Frame.rows > CAMERA_HEIGHT)
        {
            cv::resize(Frame, Frame,
                cv::Size(CAMERA_WIDTH,
                    CAMERA_HEIGHT),
                0, 0, cv::INTER_LINEAR);
        }

        {
            std::lock_guard<std::mutex>
                Lock(FrameMutex);
            LatestFrame = Frame.clone();
            bNewFrame   = true;
        }

        // Small sleep to prevent
        // CPU spinning at 100%
        std::this_thread::sleep_for(
            std::chrono::milliseconds(1));
    }
}

bool ThreadedCamera::GetLatestFrame(
    cv::Mat& Out)
{
    if (!bNewFrame) return false;
    std::lock_guard<std::mutex>
        Lock(FrameMutex);
    Out       = LatestFrame.clone();
    bNewFrame = false;
    return true;
}

void ThreadedCamera::Stop()
{
    bRunning = false;
    if (CamThread.joinable())
        CamThread.join();
    Cap.release();
}
