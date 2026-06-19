#pragma once

// ============================================
// CAMERA SETTINGS
// ============================================
#define CAMERA_WIDTH         320
#define CAMERA_HEIGHT        180
#define CAMERA_FPS            30
#define CAMERA_BUFFER_SIZE     1

// ============================================
// TRACKING SETTINGS
// ============================================
#define MAX_LOST_FRAMES       45
#define MAX_JUMP_DISTANCE    150.f
#define MIN_OBJECT_AREA      200
#define SEARCH_PADDING        40

// ============================================
// FRAME SKIP
// 3 = process every 3rd frame
// ============================================
#define FRAME_SKIP             3

// ============================================
// RED color ranges in HSV
// ============================================
#define RED_LOWER1_H   0
#define RED_LOWER1_S 120
#define RED_LOWER1_V  70
#define RED_UPPER1_H  10
#define RED_UPPER1_S 255
#define RED_UPPER1_V 255
#define RED_LOWER2_H 170
#define RED_LOWER2_S 120
#define RED_LOWER2_V  70
#define RED_UPPER2_H 180
#define RED_UPPER2_S 255
#define RED_UPPER2_V 255

// ============================================
// DISPLAY
// ============================================
#define WINDOW_NAME      "Object Tracker"
#define SHOW_DEBUG_INFO   true
#define SHOW_KALMAN_PRED  true
