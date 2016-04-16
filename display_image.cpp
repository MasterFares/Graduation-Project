




    /* *************** Stereo Camera Calibration **************************
 This code can be used to calibrate stereo cameras to get the intrinsic
 and extrinsic files.
 This code also generated rectified image, and also shows RMS Error and Reprojection error
 to find the accuracy of calibration.
 You can load saved stereo images or use this code to capture them in real time.
 Keyboard Shortcuts for real time (ie clicking stereo image at run time):
 1. Default Mode: Detecting (Which detects chessboard corners in real time)
 2. 'c': Starts capturing stereo images (With 2 Sec gap, This can be changed by changing 'timeGap' macro)
 3. 'p': Process and Calibrate (Once all the images are clicked you can press 'p' to calibrate)
 Usage: StereoCameraCallibration [params]
 --cam1 (value:0)                           Camera 1 Index
 --cam2 (value:2)                           Camera 2 Index
 --dr, --folder (value:.)                   Directory of images
 -h, --height (value:6)                     Height of the board
 --help (value:true)                        Prints this
 --images, -n (value:40)                    No of stereo pair images
 --post, --postfix (value:jpg)              Image extension. Ex: jpg,png etc
 --prefixleft, --prel (value:image_left_)   Left image name prefix. Ex: image_left_
 --prefixright, --prer (value:image_right_) Right image name postfix. Ex: image_right_
 --realtime, --rt (value:1)                 Clicks stereo images before calibration. Use if you do not have stereo pair images saved
 -w, --width (value:7)                      Width of the board
 Example:   ./stereo_calib                                              Clicks stereo images at run time.
 ./stereo_calib -rt=0 -prel=left_ -prer=right_ -post=jpg     RealTime id off ie images should be loaded from disk. With images named left_1.jpg, right_1.jpg etc.
 Cheers
 Abhishek Upperwal
 ***********************************************************************/
/* *************** License:**************************
 By downloading, copying, installing or using the software you agree to this license.
 If you do not agree to this license, do not download, install, copy or use the software.
 License Agreement
 For Open Source Computer Vision Library
 (3-clause BSD License)
 Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
 Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
 Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
 Neither the names of the copyright holders nor the names of the contributors may be used to endorse or promote products derived from this software without specific prior written permission.
 This software is provided by the copyright holders and contributors “as is” and any express or implied warranties, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed. In no event shall copyright holders or contributors be liable for any direct, indirect, incidental, special, exemplary, or consequential damages (including, but not limited to, procurement of substitute goods or services; loss of use, data, or profits; or business interruption) however caused and on any theory of liability, whether in contract, strict liability, or tort (including negligence or otherwise) arising in any way out of
 the use of this software, even if advised of the possibility of such damage.
 ************************************************** */

/* ************* Original reference:**************
 Oct. 3, 2008
 BOOK:It would be nice if you cited it:
 Learning OpenCV: Computer Vision with the OpenCV Library
 by Gary Bradski and Adrian Kaehler
 Published by O'Reilly Media, October 3, 2008
 AVAILABLE AT:
 http://www.amazon.com/Learning-OpenCV-Computer-Vision-Library/dp/0596516134
 Or: http://oreilly.com/catalog/9780596516130/
 ISBN-10: 0596516134 or: ISBN-13: 978-0596516130
 OPENCV WEBSITES:
 Homepage:      http://opencv.org
 Online docs:   http://docs.opencv.org
 Q&A forum:     http://answers.opencv.org
 Issue tracker: http://code.opencv.org
 GitHub:        https://github.com/Itseez/opencv/
 ************************************************** */


#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
// #include "opencv2/core/core.hpp"
//#include "opencv2/imgcodecs.hpp"
#include <iostream>
#include <stdlib.h>
#include <cv.h>
#include <highgui.h>
#include <stdio.h>
#include <iostream>
#include "opencv2/opencv.hpp"
#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include <sstream>

 
     using namespace std;
using namespace cv;

#include "GL/gl.h"
#include "GL/freeglut.h"

#define timeGap 3000000000U

using namespace cv;
using namespace std;

static void help() {
    cout<<"/******** HELP *******/\n";
    cout << "\nThis program helps you to calibrate the stereo cameras.\n This program generates intrinsics.yml and extrinsics.yml which can be used in Stereo Matching Algorithms.\n";
    cout<<"It also displays the rectified image\n";
    cout<<"\nKeyboard Shortcuts for real time (ie clicking stereo image at run time):\n";
    cout<<"1. Default Mode: Detecting (Which detects chessboard corners in real time)\n";
    cout<<"2. 'c': Starts capturing stereo images (With 2 Sec gap, This can be changed by changing 'timeGap' macro)\n";
    cout<<"3. 'p': Process and Calibrate (Once all the images are clicked you can press 'p' to calibrate)";
    cout<<"\nType ./stereo_calib --help for more details.\n";
    cout<<"\n/******* HELP ENDS *********/\n\n";
}

enum Modes { DETECTING, CAPTURING, CALIBRATING};
Modes mode = DETECTING;
int noOfStereoPairs;
int stereoPairIndex = 0, cornerImageIndex=0;
int goIn = 1;
Mat _leftOri, _rightOri;
int64 prevTickCount;
vector<Point2f> cornersLeft, cornersRight;
vector<vector<Point2f> > cameraImagePoints[2];
Size boardSize;

string prefixLeft;
string prefixRight;
string postfix;
string dir;

int calibType;

/////////////////////////////////////////////////////////////////////////////////

StereoBM BMState;

StereoSGBM sgbm;


/*
BMState.state->preFilterSize = 41;
BMState.state->preFilterCap = 31;
BMState.state->SADWindowSize = 41;
BMState.state->minDisparity = -64;
BMState.state->numberOfDisparities = 128;
BMState.state->textureThreshold = 5;
BMState.state->uniquenessRatio = 5;
//BMState.state->specklagewindowsize=200;
*/
////////////////////////////////////////////////////////////////////////////////


Mat displayCapturedImageIndex(Mat);
Mat displayMode(Mat);
bool findChessboardCornersAndDraw(Mat, Mat);
void displayImages();
void saveImages(Mat, Mat, int);
void calibrateStereoCamera(Size);
void calibrateInRealTime(int, int);
void calibrateFromSavedImages(string, string, string, string);



void on_HSV_Trackbar( int, void* )
{
  
}


void on_trackbar( int, void* )
{
cout<<"here"<<endl;
if ( BMState.state->preFilterSize % 2 ==0)
    BMState.state->preFilterSize++;

if(BMState.state->preFilterSize < 5)
    BMState.state->preFilterSize =5;


if ( BMState.state->SADWindowSize % 2 ==0)
    BMState.state->SADWindowSize++;

if(BMState.state->SADWindowSize < 5)
    BMState.state->SADWindowSize =5;


if(BMState.state->minDisparity > 0)
    BMState.state->minDisparity *=-1;
//while (BMState.state->numberOfDisparities %16 != 0)
 //   BMState.state->numberOfDisparities++;
}


Mat displayCapturedImageIndex(Mat img) {
    std::ostringstream imageIndex;
    imageIndex<<stereoPairIndex<<"/"<<noOfStereoPairs;
    putText(img, imageIndex.str().c_str(), Point(50, 70), FONT_HERSHEY_PLAIN, 0.9, Scalar(0,0,255), 2);
    return img;
}

Mat displayMode(Mat img) {
    String modeString = "DETECTING";
    if (mode == CAPTURING) {
        modeString="CAPTURING";
    }
    else if (mode == CALIBRATING) {
        modeString="CALIBRATED";
    }
    putText(img, modeString, Point(50,50), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,255,0), 2);
    if (mode == CAPTURING) {
        img = displayCapturedImageIndex(img);
    }
    return img;
}

bool findChessboardCornersAndDraw(Mat inputLeft, Mat inputRight) {
    _leftOri = inputLeft;
    _rightOri = inputRight;
    bool foundLeft = false, foundRight = false;
    cvtColor(inputLeft, inputLeft, COLOR_BGR2GRAY);
    cvtColor(inputRight, inputRight, COLOR_BGR2GRAY);
    foundLeft = findChessboardCorners(inputLeft, boardSize, cornersLeft, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);
    foundRight = findChessboardCorners(inputRight, boardSize, cornersRight, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);
    drawChessboardCorners(_leftOri, boardSize, cornersLeft, foundLeft);
    drawChessboardCorners(_rightOri, boardSize, cornersRight, foundRight);
    _leftOri = displayMode(_leftOri);
    _rightOri = displayMode(_rightOri);
    if (foundLeft && foundRight) {
        return true;
    }
    else {
        return false;
    }
}

void displayImages() {
    imshow("Left Image", _leftOri);
    imshow("Right Image", _rightOri);
}

void saveImages(Mat leftImage, Mat rightImage, int pairIndex) {
    cameraImagePoints[0].push_back(cornersLeft);
    cameraImagePoints[1].push_back(cornersRight);
    if (calibType == 1) {
        cvtColor(leftImage, leftImage, COLOR_BGR2GRAY);
        cvtColor(rightImage, rightImage, COLOR_BGR2GRAY);
        std::ostringstream leftString, rightString;
        leftString<<dir<<"/"<<prefixLeft<<pairIndex<<postfix;
        rightString<<dir<<"/"<<prefixRight<<pairIndex<<postfix;
        imwrite(leftString.str().c_str(), leftImage);
        imwrite(rightString.str().c_str(), rightImage);
    }
}

void calibrateStereoCamera(Size imageSize) {
    vector<vector<Point3f> > objectPoints;
    objectPoints.resize(noOfStereoPairs);
    for (int i=0; i<noOfStereoPairs; i++) {
        for (int j=0; j<boardSize.height; j++) {
            for (int k=0; k<boardSize.width; k++) {
                objectPoints[i].push_back(Point3f(float(k),float(j),0.0));
            }
        }
    }

/*

Assertion failed (nimages > 0 && nimages == (int)imagePoints1.total() && (!imgPtMat2 || nimages == (int)imagePoints2.total())) in collectCalibrationData

*/
cout<<"\n\n ------- "<<objectPoints.size() << "     ------------ - \n\n";
cout<<"\n\n ------- "<<cameraImagePoints[0].size() << "     ------------ - \n\n";
cout<<"\n\n ------- "<<cameraImagePoints[1].size() << "     ------------ - \n\n";







    Mat cameraMatrix[2], distCoeffs[2];
    cameraMatrix[0] = Mat::eye(3, 3, CV_64F);
    cameraMatrix[1] = Mat::eye(3, 3, CV_64F);
    Mat R, T, E, F;
    TermCriteria criteria = TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 100, 1e-5);

    double rms = stereoCalibrate(objectPoints, cameraImagePoints[0], cameraImagePoints[1],
                                 cameraMatrix[0], distCoeffs[0],
                                 cameraMatrix[1], distCoeffs[1],
                                 imageSize, R, T, E, F,
                                 criteria,
                                 CALIB_FIX_ASPECT_RATIO +
                                 CALIB_ZERO_TANGENT_DIST +
                                 CALIB_SAME_FOCAL_LENGTH +
                                 CALIB_RATIONAL_MODEL +
                                 CALIB_FIX_K3 + CALIB_FIX_K4 + CALIB_FIX_K5
                                  );
    cout<<"RMS Error: "<<rms<<"\n";
    double err = 0;
    int npoints = 0;
    vector<Vec3f> lines[2];
    for(int i = 0; i < noOfStereoPairs; i++ )
    {
        int npt = (int)cameraImagePoints[0][i].size();
        Mat imgpt[2];
        for(int k = 0; k < 2; k++ )
        {
            imgpt[k] = Mat(cameraImagePoints[k][i]);
            undistortPoints(imgpt[k], imgpt[k], cameraMatrix[k], distCoeffs[k], Mat(), cameraMatrix[k]);
            computeCorrespondEpilines(imgpt[k], k+1, F, lines[k]);
        }
        for(int j = 0; j < npt; j++ )
        {
            double errij = fabs(cameraImagePoints[0][i][j].x*lines[1][j][0] +
                                cameraImagePoints[0][i][j].y*lines[1][j][1] + lines[1][j][2]) +
            fabs(cameraImagePoints[1][i][j].x*lines[0][j][0] +
                 cameraImagePoints[1][i][j].y*lines[0][j][1] + lines[0][j][2]);
            err += errij;
        }
        npoints += npt;
    }
    cout << "Average Reprojection Error: " <<  err/npoints << endl;
    FileStorage fs("intrinsics.yml", FileStorage::WRITE);
    if (fs.isOpened()) {
        fs << "M1" << cameraMatrix[0] << "D1" << distCoeffs[0] <<
        "M2" << cameraMatrix[1] << "D2" << distCoeffs[1] << "R" << R << "T" << T << "E" << E << "F" << F;
        fs.release();
    }
    else
        cout<<"Error: Could not open intrinsics file.";
    Mat R1, R2, P1, P2, Q;
    Rect validROI[2];
    stereoRectify(cameraMatrix[0], distCoeffs[0], cameraMatrix[1], distCoeffs[1], imageSize, R, T, R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY, 1, imageSize, &validROI[0], &validROI[1]);
    fs.open("extrinsics.yml", FileStorage::WRITE);
    if (fs.isOpened()) {
        fs << "R" << R << "T" << T << "R1" << R1 << "R2" << R2 << "P1" << P1 << "P2" << P2 << "Q" << Q ;
        fs.release();
    }
    else
        cout<<"Error: Could not open extrinsics file";
    bool isVerticalStereo = fabs(P2.at<double>(1, 3)) > fabs(P2.at<double>(0, 3));
    Mat rmap[2][2];
    initUndistortRectifyMap(cameraMatrix[0], distCoeffs[0], R1, P1, imageSize, CV_16SC2, rmap[0][0], rmap[0][1]);
    initUndistortRectifyMap(cameraMatrix[1], distCoeffs[1], R2, P2, imageSize, CV_16SC2, rmap[1][0], rmap[1][1]);

cout<<"\n\npass 1\n\n";


    Mat canvas;
    double sf;
    int w, h;
    if (!isVerticalStereo) {
        sf = 600./MAX(imageSize.width, imageSize.height);
        w = cvRound(imageSize.width*sf);
        h = cvRound(imageSize.height*sf);
        canvas.create(h, w*2, CV_8UC3);
    }
    else {
        sf = 300./MAX(imageSize.width, imageSize.height);
        w = cvRound(imageSize.width*sf);
        h = cvRound(imageSize.height*sf);
        canvas.create(h*2, w, CV_8UC3);
    }

cout<<"\n\npass 2\n\n";



    String file;
    namedWindow("rectified");
    for (int i=0; i < noOfStereoPairs; i++) {
        for (int j=0; j < 2; j++) {
            if (j==0) {
                file = prefixLeft;
            }
            else if (j==1) {
                file = prefixRight;
            }
                        cout<<"\n\npass 3 - "<<i << "   " << j<<"\n";

            ostringstream st;
            st<<dir<<"/"<<file<<i+1<<postfix;
            Mat img = imread(st.str().c_str()), rimg, cimg;
            remap(img, rimg, rmap[j][0], rmap[j][1], INTER_LINEAR);
            cimg=rimg;



            Mat canvasPart = !isVerticalStereo ? canvas(Rect(w*j, 0, w, h)) : canvas(Rect(0, h*j, w, h));
            resize(cimg, canvasPart, canvasPart.size(), 0, 0, INTER_AREA);
            Rect vroi(cvRound(validROI[j].x*sf), cvRound(validROI[j].y*sf),
                      cvRound(validROI[j].width*sf), cvRound(validROI[j].height*sf));
            rectangle(canvasPart, vroi, Scalar(0,0,255), 3, 8);

        imshow("rectified", canvas);
       //waitKey(0);

        }



        if( !isVerticalStereo )
            for(int j = 0; j < canvas.rows; j += 16 )
                line(canvas, Point(0, j), Point(canvas.cols, j), Scalar(0, 255, 0), 1, 8);
        else
            for(int j = 0; j < canvas.cols; j += 16 )
                line(canvas, Point(j, 0), Point(j, canvas.rows), Scalar(0, 255, 0), 1, 8);
        imshow("rectified", canvas);
    }
}






















    Mat disp ,vdisp,image3D;

    Mat inputLeft, inputRight, copyImageLeft, copyImageRight , rimg , limg ;
    uchar minDisp=0;

    Mat rmap[2][2];
    Rect validROI[2];

bool isVerticalStereo ;

void loadCalibrationParameters() {
    
Mat test = imread("./image_left_1.JPEG");


    FileStorage finInterinsics("intrinsics.yml",FileStorage::READ);
    FileStorage finExtrinsics("extrinsics.yml",FileStorage::READ);

Size imageSize = test.size();

disp = Mat(imageSize.height,imageSize.width,CV_16S);
vdisp = Mat(imageSize.height,imageSize.width,CV_8U);

Mat cameraMatrix[2], distCoeffs[2];
    cameraMatrix[0] = Mat::eye(3, 3, CV_64F);
    cameraMatrix[1] = Mat::eye(3, 3, CV_64F);
    Mat R, T, E, F;


    finInterinsics["M1"]>>cameraMatrix[0];
    finInterinsics["M2"]>>cameraMatrix[1];
    finInterinsics["D1"]>>distCoeffs[0];
    finInterinsics["D2"]>>distCoeffs[1];

    cout<<endl<<"pass interinsics"<<endl;


    Mat R1, R2, P1, P2, Q;
    finExtrinsics["R" ]>>R;
    finExtrinsics["T" ]>>T;
    finExtrinsics["R1"]>>R1;
    finExtrinsics["R2"]>>R2;
    finExtrinsics["P1"]>>P1;
    finExtrinsics["P2"]>>P2;
    finExtrinsics["Q" ]>>Q;

    cout<<endl<<"pass extrinsics"<<endl;


    stereoRectify(cameraMatrix[0], distCoeffs[0], cameraMatrix[1], distCoeffs[1], imageSize, R, T, R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY, 1, imageSize, &validROI[0], &validROI[1]);
    
    cout<<endl<<"pass stereo Rectify"<<endl;

    cout<<endl<<validROI[0] <<endl;
    cout<<endl<<validROI[1] <<endl;


    isVerticalStereo = fabs(P2.at<double>(1, 3)) > fabs(P2.at<double>(0, 3));

    initUndistortRectifyMap(cameraMatrix[0], distCoeffs[0], R1, P1, imageSize, CV_16SC2, rmap[0][0], rmap[0][1]);
    initUndistortRectifyMap(cameraMatrix[1], distCoeffs[1], R2, P2, imageSize, CV_16SC2, rmap[1][0], rmap[1][1]);

    cout<<endl<<"pass undistort rectify map" <<endl;




    Mat canvas;
    double sf;
    int w, h;
    if (!isVerticalStereo) {
        sf = 600./MAX(imageSize.width, imageSize.height);
        w = cvRound(imageSize.width*sf);
        h = cvRound(imageSize.height*sf);
        canvas.create(h, w*2, CV_8UC3);
    }
    else {
        sf = 300./MAX(imageSize.width, imageSize.height);
        w = cvRound(imageSize.width*sf);
        h = cvRound(imageSize.height*sf);
        canvas.create(h*2, w, CV_8UC3);
    }




}//end of method





//read old images
/*
void rectifyInRealTime(int cam1, int cam2) {
    VideoCapture camLeft(cam1), camRight(cam2);
    if (!camLeft.isOpened() || !camRight.isOpened()) {
        cout<<"Error: Stereo Cameras not found or there is some problem connecting them. Please check your cameras.\n";
        exit(-1);
    }

namedWindow("test left");
namedWindow("test right");

Mat test = imread("./image_left_1.JPEG");

    FileStorage finInterinsics("intrinsics.yml",FileStorage::READ);
    FileStorage finExtrinsics("extrinsics.yml",FileStorage::READ);

Size imageSize = test.size();

Mat cameraMatrix[2], distCoeffs[2];
    cameraMatrix[0] = Mat::eye(3, 3, CV_64F);
    cameraMatrix[1] = Mat::eye(3, 3, CV_64F);
    Mat R, T, E, F;


    finInterinsics["M1"]>>cameraMatrix[0];
    finInterinsics["M2"]>>cameraMatrix[1];
    finInterinsics["D1"]>>distCoeffs[0];
    finInterinsics["D2"]>>distCoeffs[1];

    cout<<endl<<"pass interinsics"<<endl;


    Mat R1, R2, P1, P2, Q;
    finExtrinsics["R" ]>>R;
    finExtrinsics["T" ]>>T;
    finExtrinsics["R1"]>>R1;
    finExtrinsics["R2"]>>R2;
    finExtrinsics["P1"]>>P1;
    finExtrinsics["P2"]>>P2;
    finExtrinsics["Q" ]>>Q;

    cout<<endl<<"pass extrinsics"<<endl;


    Rect validROI[2];
    stereoRectify(cameraMatrix[0], distCoeffs[0], cameraMatrix[1], distCoeffs[1], imageSize, R, T, R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY, 1, imageSize, &validROI[0], &validROI[1]);
    
    cout<<endl<<"pass stereo Rectify"<<endl;

    cout<<endl<<validROI[0] <<endl;
    cout<<endl<<validROI[1] <<endl;


    bool isVerticalStereo = fabs(P2.at<double>(1, 3)) > fabs(P2.at<double>(0, 3));
    Mat rmap[2][2];
    initUndistortRectifyMap(cameraMatrix[0], distCoeffs[0], R1, P1, imageSize, CV_16SC2, rmap[0][0], rmap[0][1]);
    initUndistortRectifyMap(cameraMatrix[1], distCoeffs[1], R2, P2, imageSize, CV_16SC2, rmap[1][0], rmap[1][1]);

    cout<<endl<<"pass undistort rectify map" <<endl;



    Mat canvas;
    double sf;
    int w, h;
    if (!isVerticalStereo) {
        sf = 600./MAX(imageSize.width, imageSize.height);
        w = cvRound(imageSize.width*sf);
        h = cvRound(imageSize.height*sf);
        canvas.create(h, w*2, CV_8UC3);
    }
    else {
        sf = 300./MAX(imageSize.width, imageSize.height);
        w = cvRound(imageSize.width*sf);
        h = cvRound(imageSize.height*sf);
        canvas.create(h*2, w, CV_8UC3);
    }


    String file;

    Mat inputLeft, inputRight, copyImageLeft, copyImageRight , img ;
    for( int i = 0; i<40; ++i) {


        for (int j=0; j < 2; j++) {
/*            
            if (j==0) {
                file = prefixLeft;
            }
            else if (j==1) {
                file = prefixRight;
            }
                        cout<<"\n\npass 3 - "<<i << "   " << j<<"\n";

            ostringstream st;
            st<<dir<<"/"<<file<<i+1<<postfix;
            Mat rimg, cimg;
            img = imread(st.str().c_str());
            remap(img, rimg, rmap[j][0], rmap[j][1], INTER_LINEAR);
            cimg=rimg;



            Mat canvasPart = !isVerticalStereo ? canvas(Rect(w*j, 0, w, h)) : canvas(Rect(0, h*j, w, h));
            resize(cimg, canvasPart, canvasPart.size(), 0, 0, INTER_AREA);
            Rect vroi(cvRound(validROI[j].x*sf), cvRound(validROI[j].y*sf),
                      cvRound(validROI[j].width*sf), cvRound(validROI[j].height*sf));
            rectangle(canvasPart, vroi, Scalar(0,0,255), 3, 8);

            if (j==0) {
        imshow("Left Image", img);
        //imshow("test left", inputLeft);
            }
            else if (j==1) {
        imshow("Right Image", img);
        //imshow("test right",inputRight);

            }
        imshow("rectified", canvas);
       //waitKey(0);


        }



        if( !isVerticalStereo )
            for(int j = 0; j < canvas.rows; j += 16 )
                line(canvas, Point(0, j), Point(canvas.cols, j), Scalar(0, 255, 0), 1, 8);
        else
            for(int j = 0; j < canvas.cols; j += 16 )
                line(canvas, Point(j, 0), Point(j, canvas.rows), Scalar(0, 255, 0), 1, 8);
        imshow("rectified", canvas);
       waitKey(0);

        char keyBoardInput = (char)waitKey(50);
        if (keyBoardInput == 'q' || keyBoardInput == 'Q') {
            camLeft.release();
            camRight.release();
            exit(-1);
        }



    }
}

*/

































void calibrateInRealTime(int cam1, int cam2) {
    VideoCapture camLeft(cam1), camRight(cam2);
    if (!camLeft.isOpened() || !camRight.isOpened()) {
        cout<<"Error: Stereo Cameras not found or there is some problem connecting them. Please check your cameras.\n";
        exit(-1);
    }
    Mat inputLeft, inputRight, copyImageLeft, copyImageRight;
    bool foundCornersInBothImage = false;
    for( ; ; ) {
        camLeft>>inputLeft;
        camRight>>inputRight;
        if ((inputLeft.rows != inputRight.rows) || (inputLeft.cols != inputRight.cols)) {
            cout<<"Error: Images from both cameras are not of some size. Please check the size of each camera.\n";
            exit(-1);
        }
        inputLeft.copyTo(copyImageLeft);
        inputRight.copyTo(copyImageRight);
        foundCornersInBothImage = findChessboardCornersAndDraw(inputLeft, inputRight);
        if (foundCornersInBothImage && mode == CAPTURING && stereoPairIndex<noOfStereoPairs) {
            int64 thisTick = getTickCount();
            int64 diff = thisTick - prevTickCount;
            if (goIn==1 || diff >= timeGap) {
                goIn=0;
                saveImages(copyImageLeft, copyImageRight, ++stereoPairIndex);
                prevTickCount = getTickCount();
            }
        }
        displayImages();
        if (mode == CALIBRATING) {
            calibrateStereoCamera(inputLeft.size());
            camLeft.release();
            camRight.release();
            waitKey();
        }
        char keyBoardInput = (char)waitKey(50);
        if (keyBoardInput == 'q' || keyBoardInput == 'Q') {
            exit(-1);
        }
        else if(keyBoardInput == 'c' || keyBoardInput == 'C') {
            mode = CAPTURING;
        }
        else if (keyBoardInput == 'p' || keyBoardInput == 'P') {
            mode = CALIBRATING;
        }
    }
}

void calibrateFromSavedImages(string dr, string prel, string prer, string post) {
    Size imageSize;


    for (int i=0; i<noOfStereoPairs; i++) {
        Mat inputLeft, inputRight, copyImageLeft, copyImageRight;
        ostringstream imgIndex;
        imgIndex << i+1;
        bool foundCornersInBothImage = false;

        string sourceLeftImagePath, sourceRightImagePath;
        sourceLeftImagePath = dr+"/"+prel+imgIndex.str()+post;
        sourceRightImagePath = dr+"/"+prer+imgIndex.str()+post;

        inputLeft = imread(sourceLeftImagePath);
        inputRight = imread(sourceRightImagePath);
        imageSize = inputLeft.size();

cout<<"\n "<<sourceLeftImagePath<<" \n";

//imshow("test",inputLeft);


        if (inputLeft.empty() || inputRight.empty()) {
            cout<<"\nCould no find image: "<<sourceLeftImagePath<<" or "<<sourceRightImagePath<<". Skipping images.\n";
            continue;
        }
        if ((inputLeft.rows != inputRight.rows) || (inputLeft.cols != inputRight.cols)) {
            cout<<"\nError: Left and Right images are not of some size. Please check the size of the images. Skipping Images.\n";
            continue;
        }
        inputLeft.copyTo(copyImageLeft);
        inputRight.copyTo(copyImageRight);
        foundCornersInBothImage = findChessboardCornersAndDraw(inputLeft, inputRight);
        if (foundCornersInBothImage && stereoPairIndex<noOfStereoPairs) {
            saveImages(copyImageLeft, copyImageRight, ++stereoPairIndex);
        }
        else
            cout<<"\n\n error <<<<<" << sourceLeftImagePath << "--------- \n\n";
//_leftOri = inputLeft;
//_rightOri = inputRight;

      displayImages();
       waitKey(0);

    }
    if(stereoPairIndex > 2) {
        calibrateStereoCamera(imageSize);
        waitKey();
    }
    else {
        cout<<"\nInsufficient stereo images to calibrate.\n";
    }
}



    CvCapture *capture;
     
    IplImage* frame; 
    IplImage* frameCamShift; 
    IplImage* mask = cvCreateImage(cvSize(640,480),IPL_DEPTH_8U,1); 
    IplImage* frame2 = cvCreateImage(cvSize(640,480),IPL_DEPTH_8U,3);
    IplImage* imgThreshed;  // threshold for HSV settings

    CvMemStorage* storage = cvCreateMemStorage(0);
    CvMemStorage* defectstorage = cvCreateMemStorage(0);
    CvMemStorage* palmstorage = cvCreateMemStorage(0);
    CvMemStorage* fingerstorage = cvCreateMemStorage(0);
    CvSeq* fingerseq = cvCreateSeq(CV_SEQ_ELTYPE_POINT,sizeof(CvSeq),sizeof(CvPoint),fingerstorage);
     
    float radius; 
    CvPoint2D32f mincirclecenter;
    CvPoint mincirclecenter2;
           
    CvSeq* contours;
    CvSeq* hull;
    CvSeq* defect;
    CvSeq* palm = cvCreateSeq(CV_SEQ_ELTYPE_POINT,sizeof(CvSeq),sizeof(CvPoint),palmstorage); 
    //int hullcount;//hull
    CvPoint pt0,pt,p,armcenter;
    CvBox2D palmcenter,contourcenter;
         
    void getconvexhull(); 
    void fingertip(); 
    void hand(); 
     
    bool savepic = false;
 
    int palmsize[5];
    int palmsizecount=0; 
    bool palmcountfull = false;
     
    CvPoint palmposition[5];
    int palmpositioncount=0; 
    bool palmpositionfull = false;
 

bool selectObject = false;
int trackObject = 0;
Point origin;
Rect selection;
int vmin = 10, vmax = 256, smin = 30;
Mat image;





void on_opengl(void* param)
{
 glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45.0, (double)640.0 / 480.0, 0.1, 100.0);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(0, 0, 3, 0, 0, 0, 0, 1, 0);
    glClearColor(0.0, 0.0, 0.0, 0.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);



//if (mincirclecenter2 == NULL)
//    glTranslated(0.0, 0.0, -1.0);
//else
    //glTranslated(mincirclecenter2.x, mincirclecenter2.y, -1.0);
    glTranslated(0.3 + mincirclecenter2.x/1000.0, 0.3 + mincirclecenter2.y/1000.0, -1.0);

cout<<"opengl      ************************** "<<mincirclecenter2.x<< " , "<< mincirclecenter2.y<<endl;
    glRotatef( 55, 1, 0, 0 );
    glRotatef( 45, 0, 1, 0 );
    glRotatef( 0, 0, 0, 1 );
    static const int coords[6][4][3] = {
        { { +1, -1, -1 }, { -1, -1, -1 }, { -1, +1, -1 }, { +1, +1, -1 } },
        { { +1, +1, -1 }, { -1, +1, -1 }, { -1, +1, +1 }, { +1, +1, +1 } },
        { { +1, -1, +1 }, { +1, -1, -1 }, { +1, +1, -1 }, { +1, +1, +1 } },
        { { -1, -1, -1 }, { -1, -1, +1 }, { -1, +1, +1 }, { -1, +1, -1 } },
        { { +1, -1, +1 }, { -1, -1, +1 }, { -1, -1, -1 }, { +1, -1, -1 } },
        { { -1, -1, +1 }, { +1, -1, +1 }, { +1, +1, +1 }, { -1, +1, +1 } }
    };
    for (int i = 0; i < 6; ++i) {
                glColor3ub( i*20, 100+i*10, i*42 );
                glBegin(GL_QUADS);
                for (int j = 0; j < 4; ++j) {
                        glVertex3d(0.2 * coords[i][j][0], 0.2 * coords[i][j][1], 0.2 * coords[i][j][2]);
                }
                glEnd();
    }

            glFlush();

}



static void onMouse( int event, int x, int y, int, void* )
{
    if( selectObject )
    {
      /*
        selection.x = MIN(x, origin.x);
        selection.y = MIN(y, origin.y);
        selection.width = std::abs(x - origin.x);
        selection.height = std::abs(y - origin.y);
        
        selection &= Rect(0, 0, image.cols, image.rows);
        */
    }
    
    switch( event )
    {
        case CV_EVENT_LBUTTONDOWN:
            origin = Point(x,y);
            selection = Rect(x,y,0,0);
            selectObject = true;
            break;
        case CV_EVENT_LBUTTONUP:
            selectObject = false;
            if( selection.width > 0 && selection.height > 0 )
                trackObject = -1;
            break;
    }
}








     Mat hsv, hue, mask2, hist, histimg = Mat::zeros(200, 320, CV_8UC3), backproj;    

 Rect trackWindow;
    int hsize = 16;
    float hranges[] = {0,180};
    const float* phranges = hranges;
  
    
void getconvexhull()
{
  hull = cvConvexHull2( contours, 0, CV_CLOCKWISE, 0 );
  pt0 = **CV_GET_SEQ_ELEM( CvPoint*, hull, hull->total - 1 );
         
  for(int i = 0; i < hull->total; i++ )
  {
    pt = **CV_GET_SEQ_ELEM( CvPoint*, hull, i );
    //printf("%d,%d\n",pt.x,pt.y);
    cvLine( frame, pt0, pt, CV_RGB( 128, 128, 128 ),2,8,0);
    pt0 = pt;
  }
 
  defect = cvConvexityDefects(contours,hull,defectstorage); //Mʳ 
 
  for(int i=0;i<defect->total;i++)
  {
    CvConvexityDefect* d=(CvConvexityDefect*)cvGetSeqElem(defect,i);
 
    // if(d->depth < 50)
    // {
    //  p.x = d->start->x;
    //  p.y = d->start->y;
    //  cvCircle(frame,p,5,CV_RGB(255,255,255),-1,CV_AA,0);
    //  p.x = d->end->x;
    //  p.y = d->end->y;
    //  cvCircle(frame,p,5,CV_RGB(255,255,255),-1,CV_AA,0);
    //  }
    if(d->depth > 10)  
    {
      p.x = d->depth_point->x;
      p.y = d->depth_point->y;
      cvCircle(frame,p,5,CV_RGB(255,255,0),-1,CV_AA,0);
      cvSeqPush(palm,&p);
    }
 
  }
 
  //if(palm->total>1)
  //{
  //  cvMinEnclosingCircle(palm,&mincirclecenter,&radius);
  //  cvRound(radius);
  //  mincirclecenter2.x = cvRound(mincirclecenter.x);
  //  mincirclecenter2.y = cvRound(mincirclecenter.y);
  //  cvCircle(frame,mincirclecenter2,cvRound(radius),CV_RGB(255,128,255),4,8,0);
  //  cvCircle(frame,mincirclecenter2,10,CV_RGB(255,128,255),4,8,0);
  //  palmcenter =  cvMinAreaRect2(palm,0);
  //  center.x = cvRound(palmcenter.center.x);
  //  center.y = cvRound(palmcenter.center.y);
  //  cvEllipseBox(frame,palmcenter,CV_RGB(128,128,255),2,CV_AA,0);
  //  cvCircle(frame,center,10,CV_RGB(128,128,255),-1,8,0);
  //}
 
selection = cvBoundingRect(contours, 0);

/*
selection.x = MIN(x, origin.x);
        selection.y = MIN(y, origin.y);
        selection.width = std::abs(x - origin.x);
        selection.height = std::abs(y - origin.y);
        
        selection &= Rect(0, 0, image.cols, image.rows);

*/

    namedWindow( "CamShift Object Tracker", 0 );
    setMouseCallback( "CamShift Object Tracker", onMouse, 0 );
    
        Mat maskCopy(frame2);
        maskCopy.copyTo(image);
        
            cvtColor(image, hsv, CV_BGR2HSV);
            
            if( trackObject )
            {
                int _vmin = vmin, _vmax = vmax;
                
                inRange(hsv, Scalar(0, smin, MIN(_vmin,_vmax)),
                        Scalar(180, 256, MAX(_vmin, _vmax)), mask2);
                int ch[] = {0, 0};
                hue.create(hsv.size(), hsv.depth());
                mixChannels(&hsv, 1, &hue, 1, ch, 1);
                
                if( trackObject < 0 )
                {
                    Mat roi(hue, selection), maskroi(mask2, selection);
                    calcHist(&roi, 1, 0, maskroi, hist, 1, &hsize, &phranges);
                    normalize(hist, hist, 0, 255, CV_MINMAX);
                    
                    trackWindow = selection;
                    trackObject = 1;
                    
                    histimg = Scalar::all(0);
                    int binW = histimg.cols / hsize;
                    Mat buf(1, hsize, CV_8UC3);
                    for( int i = 0; i < hsize; i++ )
                        buf.at<Vec3b>(i) = Vec3b(saturate_cast<uchar>(i*180./hsize), 255, 255);
                    cvtColor(buf, buf, CV_HSV2BGR);
                    
                    for( int i = 0; i < hsize; i++ )
                    {
                        int val = saturate_cast<int>(hist.at<float>(i)*histimg.rows/255);
                        rectangle( histimg, Point(i*binW,histimg.rows),
                                  Point((i+1)*binW,histimg.rows - val),
                                  Scalar(buf.at<Vec3b>(i)), -1, 8 );
                    }
                }
                
                calcBackProject(&hue, 1, 0, hist, backproj, &phranges);

                backproj &= mask2;
                RotatedRect trackBox = CamShift(backproj, trackWindow,
                                                TermCriteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1 ));

                if( trackWindow.area() <= 1 )
                {
                    int cols = backproj.cols, rows = backproj.rows, r = (MIN(cols, rows) + 5)/6;
                    trackWindow = Rect(trackWindow.x - r, trackWindow.y - r,
                                       trackWindow.x + r, trackWindow.y + r) &
                    Rect(0, 0, cols, rows);
                }
                
                ellipse( image, trackBox, Scalar(0,0,255), 3, CV_AA );
            }
        

        
        if( selectObject && selection.width > 0 && selection.height > 0 )
        {
            Mat roi(image, selection);
            bitwise_not(roi, roi);
        }
        
        imshow( "CamShift Object Tracker", image );
        


}
 

 //
// This function is the implementation of the following thesis:
// [a method for hand gesture recognition based on morphology and fingertip-angle]
// or you can refer to my thesis which is on the dropbox folder.
//
// There's also a sample in my Dropbox folder which works on a static picture.
//

void fingertip()
{    
  int dotproduct,i;
  float length1,length2,angle,minangle,length;
  CvPoint vector1,vector2,min,minp1,minp2;
  CvPoint fingertip[20];
  CvPoint *p1,*p2,*p;
  int tiplocation[20];
  int count = 0;
  bool signal = false;
 
  //
  // p1, p, p2 forms a triangle. we calculate the angle of it to decide if it might be a fingertip or not.
  //
  for(i=0;i<contours->total;i++)
  {
    p1 = (CvPoint*)cvGetSeqElem(contours,i);
    p = (CvPoint*)cvGetSeqElem(contours,(i+20)%contours->total);
    p2 = (CvPoint*)cvGetSeqElem(contours,(i+40)%contours->total);

    if(p1 != NULL && p != NULL && p2!=NULL)
    {
    vector1.x = p->x - p1->x;
    vector1.y = p->y - p1->y;
    vector2.x = p->x - p2->x;
    vector2.y = p->y - p2->y;
    dotproduct = (vector1.x*vector2.x) + (vector1.y*vector2.y); 
    length1 = sqrtf((vector1.x*vector1.x)+(vector1.y*vector1.y));
    length2 = sqrtf((vector2.x*vector2.x)+(vector2.y*vector2.y));
    angle = fabs(dotproduct/(length1*length2));    
          
    if(angle < 0.2)
    {
      //cvCircle(frame,*p,4,CV_RGB(0,255,255),-1,8,0); 
   
      if(!signal)//
      {
        signal = true;
        min.x = p->x;
        min.y = p->y;
        minp1.x = p1->x;
        minp1.y = p1->y;
        minp2.x = p2->x;
        minp2.y = p2->y;
        minangle = angle;
      }
      else
      {
        if(angle <= minangle)
        {
          min.x = p->x;
          min.y = p->y;
          minp1.x = p1->x;
          minp1.y = p1->y;
          minp2.x = p2->x;
          minp2.y = p2->y;
          minangle = angle;
        }
      }
   
    }
 
    else//else start
    {
      if(signal)
      {
        signal = false;
        CvPoint l1,l2,l3;
        l1.x = min.x - armcenter.x;
        l1.y = min.y - armcenter.y;
 
        l2.x = minp1.x - armcenter.x;
        l2.y = minp1.y - armcenter.y;
 
        l3.x = minp2.x - armcenter.x;
        l3.y = minp2.y - armcenter.y;
 
        length = sqrtf((l1.x*l1.x)+(l1.y*l1.y));
        length1 = sqrtf((l2.x*l2.x)+(l2.y*l2.y));
        length2 = sqrtf((l3.x*l3.x)+(l3.y*l3.y));    
 
        if(length > length1 && length > length2)
        {
          //cvCircle(frame,min,6,CV_RGB(0,255,0),-1,8,0);
          fingertip[count] = min;
          tiplocation[count] = i+20;
          count = count + 1;
        }
        else if(length < length1 && length < length2)
        {
          //cvCircle(frame,min,8,CV_RGB(0,0,255),-1,8,0);
          //cvCircle(virtualhand,min,8,CV_RGB(255,255,255),-1,8,0);
          cvSeqPush(palm,&min);
          //fingertip[count] = min;
          //tiplocation[count] = i+20;
          //count = count + 1;
        }
      }
    }//else end
 }
  }//for end    
 
  for(i=0;i<count;i++)
  {
    if( (tiplocation[i] - tiplocation[i-1]) > 40)
    {
      if( fingertip[i].x >= 630  || fingertip[i].y >= 470 )
      {
        cvCircle(frame,fingertip[i],6,CV_RGB(50,200,250),-1,8,0);
      }
      else
      {
        //cvCircle(frame,fingertip[i],6,CV_RGB(0,255,0),-1,8,0);
        //cvCircle(virtualhand,fingertip[i],6,CV_RGB(0,255,0),-1,8,0);
        //cvLine(virtualhand,fingertip[i],armcenter,CV_RGB(255,0,0),3,CV_AA,0);
        cvSeqPush(fingerseq,&fingertip[i]);
      }
    }
  }
        //cvClearSeq(fingerseq);    
}
 
CvFont Font1=cvFont(3,3);
char fingersCountString[16]; // string which will contain the number of finger tips

void hand()
{
     bool useavepalm = true;
 
     if(palm->total <= 2)
     {
        useavepalm = false;            
        cvPutText(frame,"Error Palm Position!!",cvPoint(10,50),&Font1,CV_RGB(255,0,0));  
        //savepic = true;
         
        CvPoint *temp,*additional,*palmtemp;
        CvMemStorage* palm2storage = cvCreateMemStorage(0);
        CvSeq* palm2 = cvCreateSeq(CV_SEQ_ELTYPE_POINT,sizeof(CvSeq),sizeof(CvPoint),palm2storage); 
         
        for(int i=0;i<palm->total;i++)
        {
           palmtemp = (CvPoint*)cvGetSeqElem(palm,i);
                 
           for(int j=1;j<contours->total;j++)
           {
             temp =  (CvPoint*)cvGetSeqElem(contours,j);  
             if(temp->y == palmtemp->y && temp->x == palmtemp->x)
             {
                additional = (CvPoint*)cvGetSeqElem(contours,(int)(j+((contours->total)/2))%(contours->total));
                if(additional->y <= palmtemp->y)
                cvCircle(frame,*additional,10,CV_RGB(0,0,255),-1,8,0); 
                cvSeqPush(palm2,additional);                  
             }                                
           }             
        }
         
        for(int i=0;i<palm2->total;i++)
        {
           temp = (CvPoint*)cvGetSeqElem(palm2,i); 
           cvSeqPush(palm,temp);    
        }
         
        for(int i=1;i<contours->total;i++)
        {
             temp =  (CvPoint*)cvGetSeqElem(contours,1);  
             if(temp->y <= additional->y)
             additional = temp;        
        }
        cvCircle(frame,*additional,10,CV_RGB(0,0,255),-1,8,0);
        cvSeqPush(palm,additional);
                                  
     }
     
     //////////////////////////////////////////////////////////////////////////////
     cvMinEnclosingCircle(palm,&mincirclecenter,&radius);
     mincirclecenter2.x = cvRound(mincirclecenter.x);
     mincirclecenter2.y = cvRound(mincirclecenter.y);
     
    if(useavepalm){
    CvPoint avePalmCenter,distemp;
    int lengthtemp,radius2;
    avePalmCenter.x = 0;
    avePalmCenter.y = 0;
     
    for(int i=0;i<palm->total;i++) 
    {
            CvPoint *temp = (CvPoint*)cvGetSeqElem(palm,i); 
            avePalmCenter.x += temp->x;
            avePalmCenter.y += temp->y;        
    }
      
     avePalmCenter.x = (int)(avePalmCenter.x/palm->total);
     avePalmCenter.y = (int)(avePalmCenter.y/palm->total);
     radius2 = 0;
      
     for(int i=0;i<palm->total;i++)
    {
            CvPoint *temp = (CvPoint*)cvGetSeqElem(palm,i); 
            distemp.x = temp->x - avePalmCenter.x;      
            distemp.y = temp->y - avePalmCenter.y;  
            lengthtemp =  sqrtf(( distemp.x* distemp.x)+(distemp.y*distemp.y));
            radius2 += lengthtemp;
    }
     
    radius2 = (int)(radius2/palm->total);
    radius = ((0.5)*radius + (0.5)*radius2);
    mincirclecenter2.x =  ((0.5)*mincirclecenter2.x + (0.5)*avePalmCenter.x);
    mincirclecenter2.y =  ((0.5)*mincirclecenter2.y + (0.5)*avePalmCenter.y);
}
     //////////////////////////////////////////////////////////////////////////////////////
      
     palmposition[palmpositioncount].x = cvRound(mincirclecenter2.x);
     palmposition[palmpositioncount].y = cvRound(mincirclecenter2.y);
     palmpositioncount = (palmpositioncount+1)%3;
      
     if(palmpositionfull)
     {
        float xtemp=0,ytemp=0;
        for(int i=0;i<3;i++)
        {
           xtemp += palmposition[i].x;   
           ytemp += palmposition[i].y;          
        }        
         
        mincirclecenter2.x = cvRound(xtemp/3); 
        mincirclecenter2.y = cvRound(ytemp/3);    
     }
      
     if(palmpositioncount == 2 && palmpositionfull == false)
     {
        palmpositionfull = true;                       
     }
      
     cvCircle(frame,mincirclecenter2,10,CV_RGB(0,255,255),4,8,0);
     //cvCircle(virtualhand,mincirclecenter2,10,CV_RGB(0,255,255),4,8,0);
      
    
      
     ////////////////////////////////////////////////////////////////////////////////////// 
      
     palmsize[palmsizecount] = cvRound(radius);
     palmsizecount = (palmsizecount+1)%3;
      
     if(palmcountfull)
     {
        float tempcount=0;
        for(int i=0;i<3;i++)
        {
           tempcount += palmsize[i];             
        }        
         
        radius = tempcount/3;     
     }
      
     if(palmsizecount == 2 && palmcountfull == false)
     {
        palmcountfull = true;                       
     }
      
    cvCircle(frame,mincirclecenter2,cvRound(radius),CV_RGB(255,0,0),2,8,0);
    cvCircle(frame,mincirclecenter2,cvRound(radius*1.2),CV_RGB(200,100,200),1,8,0);
    //cvCircle(virtualhand,mincirclecenter2,cvRound(radius),CV_RGB(255,0,0),2,8,0);
    //cvCircle(virtualhand,mincirclecenter2,cvRound(radius*1.3),CV_RGB(200,100,200),1,8,0);
      
     //////////////////////////////////////////////////////////////////////////////////////
     
    int fingercount = 0; 
    float fingerlength;
    CvPoint tiplength,*point;
      

    for(int i=0;i<fingerseq->total;i++)
     {
         point = (CvPoint*)cvGetSeqElem(fingerseq,i);
         tiplength.x = point->x - mincirclecenter2.x;
         tiplength.y = point->y - mincirclecenter2.y;
         fingerlength = sqrtf(( tiplength.x* tiplength.x)+(tiplength.y*tiplength.y));
          
         if((int)fingerlength > cvRound(radius*1.2))
         {
                        
            fingercount += 1;   
            cvCircle(frame,*point,6,CV_RGB(0,255,0),-1,8,0);
            //sprintf ( fingersCountString, "%d", fingercount );
            //cvPutText(frame,fingersCountString,cvPoint(10,50),&Font1,CV_RGB(255,0,0));  

cout<<"detected fingers : "<<fingercount<<endl;
                     
         }        
     }
 
      
  
 
      
      
      
      
     cvClearSeq(fingerseq); 
     cvClearSeq(palm);
}



Mat out;
Mat dst;
int minH , minS , minV ;
int maxH , maxS , maxV ;

int main(int argc, char** argv) {

 namedWindow("model", CV_WINDOW_OPENGL);


glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_SINGLE);
    glutInitWindowSize(500,500);
    glutInitWindowPosition(100,100);
    //glutCreateWindow("modelx");

    setOpenGlContext("model");  

setOpenGlDrawCallback   (   "model",
on_opengl,
0 
);



    help();
    const String keys =
    "{help| |Prints this}"
    "{h height|6|Height of the board}"
    "{w width|7|Width of the board}"
    "{rt realtime|1|Clicks stereo images before calibration. Use if you do not have stereo pair images saved}"
    "{n images|40|No of stereo pair images}"
    "{dr folder|.|Directory of images}"
    "{prel prefixleft|image_left_|Left image name prefix. Ex: image_left_}"
    "{prer prefixright|image_right_|Right image name postfix. Ex: image_right_}"
    "{post postfix|jpg|Image extension. Ex: jpg,png etc}"
    "{cam1|0|Camera 1 Index}"
    "{cam2|2|Camera 2 Index}";
    //CommandLineParser parser(argc, argv, keys);
    //if(parser.has("help")) {
    //    parser.printMessage();
    //    exit(-1);
   // }
/*    boardSize = Size(parser.get<int>("w"), parser.get<int>("h"));
    noOfStereoPairs = parser.get<int>("n");
    prefixLeft = parser.get<string>("prel");
    prefixRight = parser.get<string>("prer");
    postfix = parser.get<string>("post");
    dir =parser.get<string>("dr");
    calibType = parser.get<int>("rt");*/

    boardSize = Size(6, 9);
    noOfStereoPairs = 40;
    prefixLeft = "image_left_";
    prefixRight = "image_right_";
    postfix = ".JPEG";
    dir =".";
    calibType = 1;
    int cam1 = 0;
    int cam2 = 2;


namedWindow("test");

 namedWindow("Left Image");
 namedWindow("Right Image");
namedWindow("disparity",1);


 BMState.state->preFilterSize = 21;
BMState.state->preFilterCap = 31;
BMState.state->SADWindowSize = 21;
BMState.state->minDisparity = 0;
BMState.state->numberOfDisparities = 32;
BMState.state->textureThreshold = 10;
BMState.state->uniquenessRatio = 15;
//BMState.state->specklagewindowsize=10;
BMState.state->speckleRange =32;


sgbm.SADWindowSize = 5;
sgbm.numberOfDisparities = 192;
sgbm.preFilterCap = 4;
sgbm.minDisparity = -64;
sgbm.uniquenessRatio = 1;
sgbm.speckleWindowSize = 150;
sgbm.speckleRange = 2;
sgbm.disp12MaxDiff = 10;
sgbm.fullDP = false;
sgbm.P1 = 600;
sgbm.P2 = 2400;


 //createTrackbar( "preFilterSize", "disparity", &BMState.state->preFilterSize, 255, on_trackbar ); // must be odd and within 5 - 255
  createTrackbar( "preFilterCap", "disparity", &sgbm.preFilterCap , 63, on_trackbar ); // must be within 1 - 63
 createTrackbar( "SADWindowSize", "disparity", &sgbm.SADWindowSize, 100, on_trackbar ); // must be odd, be within 5..255 and be not larger than image width or height 
 //createTrackbar( "minDisparity", "disparity", &sgbm.minDisparity , 100, on_trackbar ); 
 createTrackbar( "numberOfDisparities", "disparity", &sgbm.numberOfDisparities, 256, on_trackbar ); // must be positive and divisble by 16
 createTrackbar( "speckleWindowSize", "disparity", &sgbm.speckleWindowSize, 255, on_trackbar );
 createTrackbar( "uniquenessRatio", "disparity", &sgbm.uniquenessRatio, 100, on_trackbar );
 createTrackbar( "disp12MaxDiff", "disparity", &sgbm.disp12MaxDiff, 100, on_trackbar );
 //createTrackbar( "sgbm.fullDP", "disparity", &sgbm.fullDP, 100, on_trackbar );


 /*


 createTrackbar( "preFilterSize", "disparity", &BMState.state->preFilterSize, 255, on_trackbar ); // must be odd and within 5 - 255
  createTrackbar( "preFilterCap", "disparity", &BMState.state->preFilterCap , 63, on_trackbar ); // must be within 1 - 63
 createTrackbar( "SADWindowSize", "disparity", &BMState.state->SADWindowSize, 100, on_trackbar ); // must be odd, be within 5..255 and be not larger than image width or height 
 createTrackbar( "minDisparity", "disparity", &BMState.state->minDisparity, 100, on_trackbar ); 
 createTrackbar( "numberOfDisparities", "disparity", &BMState.state->numberOfDisparities, 256, on_trackbar ); // must be positive and divisble by 16
 createTrackbar( "textureThreshold", "disparity", &BMState.state->textureThreshold, 100, on_trackbar );
 createTrackbar( "uniquenessRatio", "disparity", &BMState.state->uniquenessRatio, 100, on_trackbar );

*/

    //switch (calibType) {
     //   case 0:
        //  calibrateFromSavedImages(dir, prefixLeft, prefixRight, postfix);

       //     break;
       // case 1:
           // calibrateInRealTime(parser.get<int>("cam1"), parser.get<int>("cam2"));
       //  calibrateInRealTime(1, 2);

         loadCalibrationParameters();

VideoCapture camLeft(cam1), camRight(cam2);
    if (!camLeft.isOpened() || !camRight.isOpened()) {
        cout<<"Error: Stereo Cameras not found or there is some problem connecting them. Please check your cameras.\n";
        exit(-1);
    }

namedWindow("test left");
namedWindow("test right");
namedWindow("depth");


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    //CvVideoWriter *writer;
     
    //capture =cvCreateFileCapture("hand4.avi") ;
    //
    //capture = cvCaptureFromCAM(0) ;

    cvNamedWindow("Webcam",0);
        cvNamedWindow("hi",0);
        cvNamedWindow("canny");

//minH = 0 , minS = 50 , minV = 60;
//maxH = 20 , maxS = 150 , maxV = 255;

minH = 0 , minS = 44 , minV = 28;
maxH = 25 , maxS = 255 , maxV = 255;


 createTrackbar( "minH", "hi", &minH, 255, on_HSV_Trackbar );
 createTrackbar( "maxH", "hi", &maxH, 255, on_HSV_Trackbar );

 createTrackbar( "minS", "hi", &minS, 255, on_HSV_Trackbar );
 createTrackbar( "maxS", "hi", &maxS, 255, on_HSV_Trackbar );

  createTrackbar( "minV", "hi", &minV, 255, on_HSV_Trackbar );
 createTrackbar( "maxV", "hi", &maxV, 255, on_HSV_Trackbar );

    //cvNamedWindow("Virtual hand",0);
    //writer = cvCreateVideoWriter("palm_output2.avi",CV_FOURCC('M','J','P','G'),15,cvSize(640,480),1);
           int frameNum = 0;

for( int i = 0; ; ++i) {

                camLeft>>inputLeft;
                inputLeft.copyTo(copyImageLeft);
                cvtColor(inputLeft, inputLeft, COLOR_BGR2GRAY);

                camRight>>inputRight;
                inputRight.copyTo(copyImageRight);
                cvtColor(inputRight, inputRight, COLOR_BGR2GRAY);
                //        cout<<"\n\npass - "<<i << "   " << j<<"\n";
            

            Mat rcimg ,lcimg;

            remap(inputRight, rimg, rmap[0][0], rmap[0][1], INTER_LINEAR);
            remap(inputLeft, limg, rmap[1][0], rmap[1][1], INTER_LINEAR);
                cout<<validROI[0]<<endl<<endl;
                cout<<validROI[1]<<endl<<endl;

            rcimg=rimg;
            lcimg=limg;

//rimg = rimg(validROI[1]);
//limg = limg(validROI[1]);

                cout<<"pass ";

            if(!isVerticalStereo)
            {
                cout<<"pass "<< i;
                //BMState( rimg, limg, disp );//;(rimg,limg,disp,CV_16S);//cvFindStereoCorrespondenceBM(rimg,limg,disp,BMState);
                sgbm( rimg, limg, disp );//;(rimg,limg,disp,CV_16S);//cvFindStereoCorrespondenceBM(rimg,limg,disp,BMState);
                disp.convertTo(vdisp, CV_8U);

                normalize(disp, vdisp, 0, 255, CV_MINMAX, CV_8U);

//disp = disp(validROI[0]);


double min, max;

Point min_loc, max_loc;

cv::minMaxLoc(vdisp, &min, &max, &min_loc, &max_loc);

                std::cout << "max disp : "<<max <<"   max location : "<<max_loc<<endl;
                
                disp.convertTo(disp, CV_8UC3);

                circle( vdisp,
                        max_loc,
                        20,
                        Scalar( 0, 0, 255 ),
                        -1,
                        8 );
            
                            imshow("disparity",vdisp);

               // reprojectImageTo3D(vdisp,image3D,Q,false,-1);
               // imshow("depth",image3D);
cout<<" - done\n";
            }//end of if


            //Mat canvasPart = !isVerticalStereo ? canvas(Rect(w*j, 0, w, h)) : canvas(Rect(0, h*j, w, h));
            //resize(rcimg, canvasPart, canvasPart.size(), 0, 0, INTER_AREA);
            //Rect vroi(cvRound(validROI[j].x*sf), cvRound(validROI[j].y*sf),
            //          cvRound(validROI[j].width*sf), cvRound(validROI[j].height*sf));
            //rectangle(canvasPart, vroi, Scalar(0,0,255), 3, 8);

        imshow("Left Image", limg);
        imshow("test left", copyImageLeft);

        imshow("Right Image", rimg);
        imshow("test right",copyImageRight);

        //imshow("rectified", canvas);
       //waitKey(0);



        /*
        if( !isVerticalStereo )
            for(int j = 0; j < canvas.rows; j += 16 )
                line(canvas, Point(0, j), Point(canvas.cols, j), Scalar(0, 255, 0), 1, 8);
        else
            for(int j = 0; j < canvas.cols; j += 16 )
                line(canvas, Point(j, 0), Point(j, canvas.rows), Scalar(0, 255, 0), 1, 8);
        imshow("rectified", canvas);
*/
        char keyBoardInput = (char)waitKey(50);
        if (keyBoardInput == 'q' || keyBoardInput == 'Q') {
            camLeft.release();
            camRight.release();
            exit(-1);
        



    }//end of if



////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//Mat to iplimage
frame = cvCreateImage(cvSize(copyImageRight.cols,copyImageRight.rows),8,3);
IplImage ipltemp=copyImageRight;
cvCopy(&ipltemp,frame);


        cvCopy(frame,frame2);



        //cvWriteFrame(writer,frame);
        cvCvtColor(frame,frame,CV_BGR2HSV); 

        cvInRangeS(frame, Scalar(minH, minS, minV), Scalar(maxH, maxS, maxV), mask); //skin
        //cvInRangeS(frame, Scalar(20, 0, 0,0), Scalar(178, 200, 50,0), mask);   //black gloves


        cvCvtColor(frame,frame,CV_HSV2BGR);
        Mat frameCopy = Mat(mask);
        medianBlur ( frameCopy, frameCopy, 7 );
        cvErode(mask,mask,0,2); //ERODE first then DILATE to eliminate the noises.
        cvDilate(mask,mask,0,2);
         cvShowImage("hi",mask);

 
        cvFindContours( mask, storage, &contours, sizeof(CvContour),
                   CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, cvPoint(0,0) );
 

        // We choose the first contour in the list which is longer than 650.
        // You might want to change the threshold to which works the best for you.
        while(contours && contours->total <= 650)
        {
          contours = contours->h_next;
        }
 
//Mat can(frame);

/// Detect edges using canny
//  Canny( can, can, 80, 80*2, 3 );


 //       imshow("canny",can);


    cvDrawContours( frame, contours, CV_RGB(100,100,100), CV_RGB(0,255,0), 1, 2, CV_AA, cvPoint(0,0) );
 
        //
        // Use a rectangle to cover up the contour.
        // Find the center of the rectangle (armcenter). Fingertip() needs it.
        //
        if(contours)
        {
          contourcenter =  cvMinAreaRect2(contours,0);
          armcenter.x = cvRound(contourcenter.center.x);
          armcenter.y = cvRound(contourcenter.center.y);
          //cvCircle(frame,armcenter,10,CV_RGB(255,255,255),-1,8,0);
          getconvexhull();
          fingertip();
          hand();
        }
 
 
        cvShowImage("Webcam",frame);
        updateWindow("model");
         
        if(cvWaitKey(1)>=0 || !frame)
        {
              //cvSaveImage("normal.jpg",frame2);
              break;
        }
    }       
    cvDestroyWindow("Webcam");
    //cvDestroyWindow("Virtual hand");
    //cvReleaseVideoWriter(&writer);

       //     break;
       // default:
       //     cout<<"-rt should be 0 or 1. Ex: -rt=1\n";
        //    break;
    //}
    return 0;
}
