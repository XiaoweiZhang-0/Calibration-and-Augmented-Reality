/*
  Xiaowei Zhang
  23SP
  **/
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <dirent.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <sys/stat.h>
#include "cv_utils.h"
using namespace std;
using namespace cv;


// 
//1.the first user input should be the path to the chessboard image
int main(int argc, char *argv[]){
    cout<<"Initiating Calibration Procedure..."<<endl;
    std::vector<std::vector<cv::Vec3f> > pointList;
    std::vector<std::vector<cv::Point2f> > cornerList;
    cv::Mat chessboard;

    mkdir("Data", 0777);
    const cv::Size patternSize = cv::Size(9, 6);
    std::vector<cv::Vec3f> pointSet;
    std::vector<cv::Point2f> cornerSet;

    ifstream f("calibration.yml");
    if(!f.good()){
      /**
       * task 2
      */
      task2(cornerSet, pointSet, pointList, cornerList, patternSize, chessboard);

      /**
       * task 3
       **/
      task3(chessboard, pointList, cornerList);
    }

    


    /**
     * Task 4
     * **/
    task4();

    /**
     * Task 7
    */
    task7();

    return 0;
}