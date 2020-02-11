// 3DMeasure.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/video.hpp>
#include <opencv2/calib3d.hpp>

#include <iostream>
#include <string>
#include "BoardGen.h"

using namespace cv;

static bool readCameraParameters(std::string filename, cv::Mat& camMatrix, Mat& distCoeffs) {
    FileStorage fs(filename, FileStorage::READ);
    if (!fs.isOpened())
        return false;
    fs["camera_matrix"] >> camMatrix;
    fs["distortion_coefficients"] >> distCoeffs;
    return true;
}

int main()
{
    VideoCapture cap;

    cv::Mat cameraMatrix, distCoeffs;
    // camera parameters are read from somewhere
    readCameraParameters("resources/out_camera_data.xml", cameraMatrix, distCoeffs);


    // open the default camera, use something different from 0 otherwise;
    // Check VideoCapture documentation.
    if (!cap.open(0))
        return 0;

    Board* b = new Board();
    Board* b2 = new Board();


    imwrite("board1.png", b->boardImage);
    imwrite("board2.png", b2->boardImage);

    Mat lastFrame;
    Vec3d lrvec,ltvec;

    bool detectedLast = false;

    for(;;)
    {
        Mat frame;
        cap >> frame;
        if (frame.empty()) break; // end of video stream
        Mat out;
        frame.copyTo(out);
        Vec3d rvec, tvec;
        bool detected = b->detect(frame, cameraMatrix, distCoeffs, rvec, tvec, out);

        if (detectedLast) {
            cv::Vec3d transposedR, transposedT;
            cv::transpose(lrvec, transposedR);
            cv::transpose(ltvec, transposedT);

            cv::Mat R1, R2, P1, P2, Q;

            cv::Ptr<cv::StereoBM> stereo = cv::StereoBM::create();
            cv::Size imgSize = cv::Size(frame.cols, frame.rows);
            cv::stereoRectify(cameraMatrix, distCoeffs, cameraMatrix, distCoeffs, imgSize, rvec.mul(transposedR), tvec.mul(transposedT), R1, R2, P1, P2, Q);

            cv::Mat map1[2], map2[2];

            cv::initUndistortRectifyMap(cameraMatrix, distCoeffs, R1, P1, imgSize, CV_16SC2, map1[0], map2[0]);
            cv::initUndistortRectifyMap(cameraMatrix, distCoeffs, R2, P2, imgSize, CV_16SC2, map1[1], map2[1]);



        }

        imshow("Output", out);
        if (waitKey(1) == 27) break;

        if (detected) {
            lastFrame = frame;
            lrvec = rvec;
            ltvec = tvec;
            //detectedLast = true;
        }
        else {
            detectedLast = false;
        }
    }
    // the camera will be closed automatically upon exit
    // cap.close();
    return 0;
}
