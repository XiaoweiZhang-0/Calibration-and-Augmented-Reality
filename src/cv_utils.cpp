/**
 * Xiaowei Zhang
 * 23SP
*/
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <dirent.h>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <iostream>
#include <fstream>
#include <sys/stat.h>
#include "cv_utils.h"
using namespace std;
using namespace cv;

void task3(cv::Mat chessboard, std::vector<std::vector<cv::Vec3f> > pointList, std::vector<std::vector<cv::Point2f> > cornerList){
    cout<<"--------------------------------"<<endl;
    cout<<"Initiating Task 3..."<<endl;
    cv::Mat cameraMatrix =  Mat::eye(3, 3, CV_64FC1);
    // cameraMatrix.at<double>(0,2) = (chessboard.cols)/2;
    // cameraMatrix.at<double>(1,2) = (chessboard.rows/2);
    cameraMatrix.at<double>(0,2) = 1080/2;
    cameraMatrix.at<double>(1,2) = 1920/2;
    cv::Mat distCoeffs = Mat::zeros(8, 1, CV_64FC1);
    ofstream myfile;
    cv::TermCriteria crit = cv::TermCriteria(TermCriteria::EPS,  100, 1);
    std::vector<cv::Mat> rvecs;
    std::vector<cv::Mat> tvecs;


    myfile.open("report.txt");
    cv::Size imageSize= chessboard.size();
    cout<<"Camera Matrix: "<<endl<<cameraMatrix<<endl;
    cout<<"Distortion Coefficients: "<<endl<<distCoeffs<<endl;
    cout<<"Calibrating..."<<endl;
    // cout<<cornerList.size()<<" "<<pointList.size()<<endl;
    double error = cv::calibrateCamera(pointList, cornerList, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs, CALIB_FIX_ASPECT_RATIO|CALIB_RATIONAL_MODEL, crit);
    cout<<"Camera Matrix: "<<endl<<cameraMatrix<<endl;
    cout<<"Distortion Coefficients: "<<endl<<distCoeffs<<endl;
    myfile<<"Re-projection Error: "<<error<<endl;
    myfile.close();
       // Save calibration parameters to a YAML file
    std::string outputFile = "calibration.yml";
    cv::FileStorage fs(outputFile, cv::FileStorage::WRITE);

    if (!fs.isOpened())
    {
        std::cout << "Error: Could not open the output file." << std::endl;
        return;
    }

    fs << "camera_matrix" << cameraMatrix;
    fs << "distortion_coefficients" << distCoeffs;

    fs.release();

    std::cout << "Camera calibration parameters saved to: " << outputFile << std::endl;

    for(int i=0; i<rvecs.size();i++){
      myfile.open("Data/"+std::to_string(i+1)+"/data.txt");
      myfile<<"Rotations:"<<endl;
      myfile<<rvecs[i]<<endl;
      myfile<<"Translations:"<<endl;
      myfile<<tvecs[i]<<endl;
      myfile.close();
    }
    cout<<"Task 3 completed."<<endl;
}

void task4(){
    cout<<"--------------------------------"<<endl;
    cout<<"Initiating Task 4..."<<endl;
    vector<cv::Point3f> pointSet = {{1, -1, 0}, {2, -1, 0}, {3, -1, 0}, {4,-1,0}, {5,-1,0}, {6,-1,0}, {7,-1,0}, {8,-1,0}, {9,-1,0},
      {1, -2, 0}, {2, -2, 0}, {3, -2, 0}, {4,-2,0}, {5,-2,0}, {6,-2,0}, {7,-2,0}, {8,-2,0}, {9,-2,0},
      {1, -3, 0}, {2, -3, 0}, {3, -3, 0}, {4,-3,0}, {5,-3,0}, {6,-3,0}, {7,-3,0}, {8,-3,0}, {9,-3,0},
      {1, -4, 0}, {2, -4, 0}, {3, -4, 0}, {4,-4,0}, {5,-4,0}, {6,-4,0}, {7,-4,0}, {8,-4,0}, {9,-4,0},
      {1, -5, 0}, {2, -5, 0}, {3, -5, 0}, {4,-5,0}, {5,-5,0}, {6,-5,0}, {7,-5,0}, {8,-5,0}, {9,-5,0},
      {1, -6, 0}, {2, -6, 0}, {3, -6, 0}, {4,-6,0}, {5,-6,0}, {6,-6,0}, {7,-6,0}, {8,-6,0}, {9,-6,0}};
    cv::Size patternSize = cv::Size(9, 6);
    std::vector<cv::Point3f> outerCorners = {{0,0,0}, {10, 0, 0}, {10, -7, 0}, {0, -7, 0}, {0, 0, 0}};
    std::vector<cv::Point3f> points = {
        {1,-1,0}, {1,-2,0}, {1,-3, 0}, {1,-4,0},{2,-5,0}, {3,-6,0}, {4,-6,0},{5,-5,0}, {6,-4,0}, {6,-3,0},{7,-3,0}, {8,-4,0}, {8,-5,0}, {8,-6,0}, {9,-6,0}, {9,-5,0}, {9,-4,0}, {9,-3,-0}, {8,-2,0}, {7,-1,0}, {6,-1,0},
        {5,-2,0}, {5,-3,0}, {4,-4,0}, {3,-4,0}, {2,-3,0}, {2,-2,0}, {2,-1,0}, {1,-1,0}, {1,-1,3}, {1,-2,3}, {1,-3,3},
        {1,-4,3}, {2,-5,3}, {3,-6,3}, {4,-6,3}, {5,-5,3}, {6,-4,3}, {6,-3,3}, {7,-3,3}, {8,-4,3}, {8,-5,3}, {8,-6,3}, {9,-6,3}, {9,-5,3}, {9,-4,3}, {9,-3,3}, {8,-2,3}, {7,-1, 3}, {6,-1,3},
        {5,-2,3}, {5,-3,3}, {4,-4,3},{3,-4,3}
    };
    std::vector<cv::Point2f> cornerProjects;
    std::vector<cv::Point2f> pointProjects;
    //Read calibration parameters from a YAML file
    std::string calibrationFile = "calibration.yml";
    cv::FileStorage fs1(calibrationFile, cv::FileStorage::READ);

    if (!fs1.isOpened())
    {
        std::cout << "Error: Could not open the calibration file." << std::endl;
        return;
    }

    cv::Mat cameraMatrix, distCoeffs;
    fs1["camera_matrix"] >> cameraMatrix;
    fs1["distortion_coefficients"] >> distCoeffs;

    // cv::VideoCapture cap("IMG_1284.MOV");
    cout<<"Please specify a source for the input of video stream, it can be a path to a preexisting video file, or by default press '0' for camera"<<endl;
    string in;
    cin>>in;
    cv::VideoCapture cap;
    if(in.compare("0")==0){
        cap.open(0);
    }
    else{
        cap.open(in);
    }

     // Create a window to display the video
    cv::namedWindow("Video Loop", cv::WINDOW_AUTOSIZE);

    // Check if the video file is opened successfully
    if (!cap.isOpened())
    {
        std::cout << "Error: Could not open the video file." << std::endl;
        return;
    }
        // Create a window to display the video
    cv::namedWindow("Video Loop", cv::WINDOW_AUTOSIZE);
    bool saved = false;
   // Get the video resolution
    int frameWidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frameHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

    // Set up the VideoWriter object
    cv::VideoWriter writer("task6.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, cv::Size(frameWidth, frameHeight));
    while (true)
    {
        cv::Mat frame, grayFrame;
        cap >> frame; // Read a new frame from the webcam

        // Break the loop if we reached the end of the video
        if (frame.empty())
        {
            break;
        }
        // cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);
        // std::vector<cv::Point2f> imagePoints;
        vector<cv::Point2f> cornerSet;
        bool found = cv::findChessboardCorners(frame, patternSize, cornerSet, 0);
        if(found){
            cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);
            // Refine corner positions
            cv::cornerSubPix(grayFrame, cornerSet, cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.001));

            // Estimate chessboard pose
            cv::Mat rvec, tvec;
            bool success = cv::solvePnP(pointSet, cornerSet, cameraMatrix, distCoeffs, rvec, tvec);

            if (success)
            {
                // Print the rotation and translation vectors
                std::cout << "Rotation vector: " << rvec.t() << std::endl;
                std::cout << "Translation vector: " << tvec.t() << std::endl;
                cv::projectPoints(outerCorners, rvec, tvec, cameraMatrix, distCoeffs, cornerProjects);
                cv::projectPoints(points, rvec, tvec, cameraMatrix, distCoeffs, pointProjects);
                // for(int i =0 ;i<cornerProjects.size();i++ ){
                //     cout<<cornerProjects[i]<<endl;
                // }
                // break;
                // cout<<cornerProjects.size()<<endl;
                // std::vector<cv::Point2f> test = {cornerProjects[0]};
                // vector<cv::Point2f>::const_iterator first = pointProjects.end()-4;
                // vector<cv::Point2f>::const_iterator second = pointProjects.end();
                // vector<cv::Point2f> sub(first, second);
                drawChessboardCorners(frame, patternSize, pointProjects, found);
                // drawChessboardCorners(frame, patternSize, sub, found);
                cv::Scalar color(0, 0, 255);
                for(int i=0; i<cornerProjects.size(); i++){
                  cv::circle(frame, cornerProjects[i], 20, color);
                }              
            }

        }
        cv::imshow("Video Loop - Chessboard Pose", frame);
        writer.write(frame);
        // if(!saved){
        //   imwrite("task5.jpeg", frame);
        //   saved = true;
        // }
        // Wait for a key press and break the loop if the 'q' key is pressed
        char key = static_cast<char>(cv::waitKey(30)); // Wait for 30 ms
        if (key == 'q' || key == 27) // 27 is the 'Esc' key
        {
            break;
        }
    }
    // Release the VideoCapture object and close the display window
    cap.release();
    writer.release();
    cv::destroyAllWindows();
    cout<<"Task 4 completed."<<endl;
}

void task2(std::vector<cv::Point2f>& cornerSet, std::vector<cv::Vec3f>& pointSet, std::vector<std::vector<cv::Vec3f>>& pointList, std::vector<std::vector<cv::Point2f>>& cornerList, const cv::Size patternSize, cv::Mat& chessboard){
    cout<<"--------------------------------"<<endl;
    cout<<"Initiating Task 2..."<<endl;
    int count = 0;
    int offset = 0;
    while(true){
      cout<<"Please specify an image path for calibration, or Enter 'end' to halt"<<endl;
        string userIn;
      cin>>userIn;
      if(userIn.compare("end")==0){
        break;
      }
      const int winSize = 11;
      int found;
      // strcpy(userIn, argv[1]);
      // std::cout<<imagePath<<std::endl;
    //   if(count >= images.size()){
    //     break;
    //   }
    //   string userIn = images[count];
      cout<<"image file: "<<userIn<<endl;
      chessboard= cv::imread(userIn);
      count++;
      found = cv::findChessboardCorners(chessboard, patternSize, cornerSet, 0);
      if(found){
        Mat chessboardGray;
        cvtColor(chessboard, chessboardGray, COLOR_BGR2GRAY);
        cornerSubPix(chessboardGray, cornerSet, Size(winSize,winSize),
                  Size(-1,-1), TermCriteria( TermCriteria::EPS+TermCriteria::COUNT, 30, 0.0001 ));
        drawChessboardCorners(chessboard, patternSize, Mat(cornerSet), found);
        // cv::imshow("chessboard with corners", chessboard);
        // waitKey(0);
        // cout<<"Image_with_corners/"+userIn.substr(0, userIn.find("."))+"_with_corners.png"<<endl;
        string imageName = userIn.substr(userIn.find("/")+1, userIn.find(".")-userIn.find("/")-1);
        string dirName = "Data/"+std::to_string(count-offset);
        mkdir(dirName.c_str(), 0777);
        cv::imwrite((dirName + "/"+imageName+"_with_corners.png"), chessboard);
        int numOfCorners = cornerSet.size();
        cout<<numOfCorners<<" corners found."<<endl;
        cout<<"The first corner has coordinates "<<cornerSet[0]<<endl;
      }
      else{
        cout<<"Failed to find inner corners"<<endl;
        // cout<<"image file: "<<userIn<<endl;
        offset++;
        continue;
      }
      // for(int i=0; i<numOfCorners; i++){
      //   cout<<cornerSet[i]<<endl;
      // }
      cout<<"Do you want to save the most recent found corners? Press 's' to save, press otherwise not to save"<<endl;
      string response;
      cin>>response;
      if(response.compare("s")==0){
        cornerList.push_back(cornerSet);
      cout<<"If the image you used is different from the default checkerboard, please enter the inner corner coordinates, in the form of {[x1,y1,z1],...,[x2,y2,z2]}, else Press 'n'"<<endl;
      string input;
      cin>>input;
      if(input.compare("n")!=0){
        input = input.substr(1,input.size()-2);

        size_t pos = 0;
        std::string token;
        string delimiter = ",";
        cv::Vec3f temp;
        while((pos = input.find(delimiter)) != std::string::npos){
          token = input.substr(0, pos);
          cout<<token.substr(1,1).c_str()<<endl;
          double x = atof(token.substr(1,1).c_str());
          double y = atof(token.substr(3,1).c_str());
          double z = atof(token.substr(5,1).c_str());
          temp[0] = x;
          temp[1] = y;
          temp[2] = z;
          pointSet.push_back(temp);
          input.erase(0, pos + delimiter.length());
        }
      }
      else{
      //   pointSet = {{1,-6,0}, {1,-5, 0}, {1,-4,0}, {1,-3,0}, {1,-2,0},{1,-1,0}, {2,-6,0}, {2,-5,0}, {2,-4,0}, {2,-3,0}, {2,-2,0}, {2,-1,0},{3,-6,0}, {3,-5,0}, {3,-4,0}, {3,-3,0}, {3,-2,0}, {3,-1,0}, {4,-6,0}, {4,-5,0}, {4,-4,0}, {4,-3,0}, {4,-2,0}, {4,-1,0}, {5,-6,0}, {5,-5,0}, {5,-4,0}, {5,-3,0}, {5,-2,0}, {5,-1,0}, {6,-6,0}, {6,-5,0}, {6,-4,0}, {6,-3,0}, {6,-2,0}, {6,-1,0}, {7,-6,0}, {7,-5,0}, {7,-4,0}, {7,-3,0}, {7,-2,0}, {7,-1,0}, {8,-6,0}, {8,-5,0}, {8,-4,0}, {8,-3,0}, {8,-2,0}, {8,-1,0}, {9,-6,0}, {9,-5,0}, {9,-4,0}, {9,-3,0}, {9,-2,0}, {9,-1,0}};
      // }
      // pointSet = {{1,-6,0}, {1,-5, 0}, {1,-4,0}, {1,-3,0}, {1,-2,0},{1,-1,0}, {2,-6,0}, {2,-5,0}, {2,-4,0}, {2,-3,0}, {2,-2,0}, {2,-1,0},{3,-6,0}, {3,-5,0}, {3,-4,0}, {3,-3,0}, {3,-2,0}, {3,-1,0}, {4,-6,0}, {4,-5,0}, {4,-4,0}, {4,-3,0}, {4,-2,0}, {4,-1,0}, {5,-6,0}, {5,-5,0}, {5,-4,0}, {5,-3,0}, {5,-2,0}, {5,-1,0}, {6,-6,0}, {6,-5,0}, {6,-4,0}, {6,-3,0}, {6,-2,0}, {6,-1,0}, {7,-6,0}, {7,-5,0}, {7,-4,0}, {7,-3,0}, {7,-2,0}, {7,-1,0}, {8,-6,0}, {8,-5,0}, {8,-4,0}, {8,-3,0}, {8,-2,0}, {8,-1,0}, {9,-6,0}, {9,-5,0}, {9,-4,0}, {9,-3,0}, {9,-2,0}, {9,-1,0}};
        pointSet = {{1, -1, 0}, {2, -1, 0}, {3, -1, 0}, {4,-1,0}, {5,-1,0}, {6,-1,0}, {7,-1,0}, {8,-1,0}, {9,-1,0},
        {1, -2, 0}, {2, -2, 0}, {3, -2, 0}, {4,-2,0}, {5,-2,0}, {6,-2,0}, {7,-2,0}, {8,-2,0}, {9,-2,0},
        {1, -3, 0}, {2, -3, 0}, {3, -3, 0}, {4,-3,0}, {5,-3,0}, {6,-3,0}, {7,-3,0}, {8,-3,0}, {9,-3,0},
        {1, -4, 0}, {2, -4, 0}, {3, -4, 0}, {4,-4,0}, {5,-4,0}, {6,-4,0}, {7,-4,0}, {8,-4,0}, {9,-4,0},
        {1, -5, 0}, {2, -5, 0}, {3, -5, 0}, {4,-5,0}, {5,-5,0}, {6,-5,0}, {7,-5,0}, {8,-5,0}, {9,-5,0},
        {1, -6, 0}, {2, -6, 0}, {3, -6, 0}, {4,-6,0}, {5,-6,0}, {6,-6,0}, {7,-6,0}, {8,-6,0}, {9,-6,0}};
      }
      pointList.push_back(pointSet);
      }
      // cv::Point2f first;
      // cv::Point2f second = cornerSet[1];
      if(cornerList.size()<5){
        cout<<"A minimum of 5 calibration frames is required to calibratie the camera, current count is "<<cornerList.size()<<" , please include more calibration images."<<endl;
      }
      else{
        cout<<"Minimum requirements for calibration are fulfilled, do you want to start calibration?"<<endl;
        cout<<"Press 'Y' for Calibration"<<endl;
        string ans;
        cin>>ans;
        if(ans.compare("Y")==0){
          break;
        }
      }
    }
    cout<<"Task 2 completed."<<endl;
}

void task7(){
    cout<<"-----------------------"<<endl;
    cout<<"Initiating Task 7..."<<endl;
    cv::VideoCapture cap(1);
        if (!cap.isOpened()) {
        std::cerr << "Error: Unable to open the camera" << std::endl;
    }

    // Create a SURF detector with a threshold
    double hessianThreshold = 200;
    cv::Ptr<cv::xfeatures2d::SURF> surf = cv::xfeatures2d::SURF::create(hessianThreshold);
       // Get the video resolution
    int frameWidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frameHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

    // Set up the VideoWriter object
    cv::VideoWriter writer("task7.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 60, cv::Size(frameWidth, frameHeight));
    
    // bool saved = false;
    while (true) {
        // Capture a frame from the camera
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "Error: Unable to capture a frame" << std::endl;
            break;
        }

        // Convert the frame to grayscale
        cv::Mat grayFrame;
        cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);

        // Detect SURF keypoints
        std::vector<cv::KeyPoint> keypoints;
        surf->detect(grayFrame, keypoints);

        // Draw the keypoints on the original frame
        cv::Mat outputFrame;
        cv::drawKeypoints(frame, keypoints, outputFrame, cv::Scalar(0, 255, 0), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

        // Show the output frame
        cv::imshow("SURF Features", outputFrame);
        // if(!saved){
        //     cv::imwrite("task7.jpeg", outputFrame);
        // }
        writer.write(outputFrame);
        // Break the loop if the user presses the 'ESC' key
        if (cv::waitKey(1) == 27) {
            break;
        }
    }

    // Release the camera and destroy the window
    cap.release();
    writer.release();
    cv::destroyAllWindows();

    cout<<"Task 7 completed"<<endl;
}