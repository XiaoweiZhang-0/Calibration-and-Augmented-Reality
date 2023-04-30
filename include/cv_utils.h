#ifndef CVS_UTIL_H
#define CVS_UTIL_H

void task3(cv::Mat chessboard, std::vector<std::vector<cv::Vec3f> > pointList, std::vector<std::vector<cv::Point2f> > cornerList);

void task4();

void task2(std::vector<cv::Point2f>& cornerSet, std::vector<cv::Vec3f>& pointSet, std::vector<std::vector<cv::Vec3f>>& pointList, std::vector<std::vector<cv::Point2f>>& cornerList, const cv::Size patternSize, cv::Mat& chessboard);

void task7();
#endif