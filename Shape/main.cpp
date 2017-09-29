#define _USE_MATH_DEFINES

#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"

#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <vector>
#include <limits>
#include <fstream>
#include <cmath>
#include <math.h>
#include <sstream>
#include <iostream>
#include <string>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <time.h>
#include <cstdlib>
#include <ctime>

#include "shape.h"

using namespace cv;
using namespace std;
using namespace Eigen;

//void Shape::MatType(Mat inputMat)
//{
//	// Helper function to indentify cv::Mat type and how to access single element.
//	int inttype = inputMat.type();
//
//	string r, a;
//	uchar depth = inttype & CV_MAT_DEPTH_MASK;
//	uchar chans = 1 + (inttype >> CV_CN_SHIFT);
//	switch (depth) {
//	case CV_8U:  r = "8U";   a = "Mat.at<uchar>(y,x)"; break;
//	case CV_8S:  r = "8S";   a = "Mat.at<schar>(y,x)"; break;
//	case CV_16U: r = "16U";  a = "Mat.at<ushort>(y,x)"; break;
//	case CV_16S: r = "16S";  a = "Mat.at<short>(y,x)"; break;
//	case CV_32S: r = "32S";  a = "Mat.at<int>(y,x)"; break;
//	case CV_32F: r = "32F";  a = "Mat.at<float>(y,x)"; break;
//	case CV_64F: r = "64F";  a = "Mat.at<float>(y,x)"; break;
//	default:     r = "User"; a = "Mat.at<UKNOWN>(y,x)"; break;
//	}
//	r += "C";
//	r += (chans + '0');
//	cout << "Mat is of type " << r << " and should be accessed with " << a << endl;
//
//}

Mat Shape::setRoi(Mat input, int roiSize, int x, int y){

	//###############################################################################
	// FUNCTION: 
	// Sets a region of interest (ROI) by using a square with <int roiSize> as width
	// and height to crop out a region from original image.
	//
	// INPUT: 
	// cv::Mat input: input image as OpenCV matrix
	// int roiSize: desired size of ROI
	// int x: x-coordinate of upper left corner of ROI
	// int y: y-coordinate of upper left corner of ROI
	//
	// OUTPUT:
	// cv::Mat output: cropped image
	//###############################################################################
	
	//*******************************************************************************
	// Variables
	Mat output;
	Rect roi;
	
	//*******************************************************************************
	// Setting ROI
	roi = Rect(x, y, roiSize, roiSize);
	output = input(roi);

	return output;
}

vector<vector<Point> > Shape::findCont(Mat input, int roiSize){

	//###############################################################################
	// FUNCTION: 
	// Using cv::findContours to find shapes (ridges of skin microrelief) in image. 
	// Shapes that are adjacent to one of the four borders of the image are
	// incomplete shapes and are filtered out. The thickness of the border area at
	// which shapes are filtered out can be adjusted with <int border>.
	//
	// INPUT: 
	// cv::Mat input: input image as OpenCV matrix
	// int roiSize: size of the image patch
	//
	// OUTPUT:
	// vector<vector<Point>> inliers: a list of every pixel on the contour for each detected 
	// shape as (x,y) coordinates  
	//###############################################################################

	//*******************************************************************************
	// Variables
	Mat in;
	vector<vector<Point> > contours;
	vector<vector<Point> > inliers;
	vector<int> outIndex;
	int border = 1;
	
	//*******************************************************************************
	// Copying input 
	input.copyTo(in);

	//*******************************************************************************
	// Finding Contours function to extract shapes (see OpenCV documentation)
	findContours(in, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

		//*******************************************************************************
		// Detecting incomplete shapes. Shapes that touch the border-band with width 
		// <int border>. Indices of those outliers are stored in <vector<int> outIndex>.
		for (size_t i = 0; i < contours.size(); i++){
			for (size_t j = 0; j < contours[i].size(); j++){
				
				if ((contours[i][j].x <= border) || (contours[i][j].x >= roiSize - border) || 
					(contours[i][j].y <= border) || (contours[i][j].y >= roiSize - border) ){

					outIndex.push_back(i);
				}
			} 
		}

		//*******************************************************************************
		// Storing all shapes, whose indices do not appear on the list of outliers stored
		// in <vector<int> outIndex>. 
		for (size_t i = 0; i < outIndex.size() - 1; i++){
			if ((outIndex[i + 1] - outIndex[i]) != 1){
				int delta = outIndex[i + 1] - outIndex[i];

				for (int j = 0; j < delta - 1; j++){
					inliers.push_back(contours[outIndex[i] + j + 1]);
				}
			}
		}

	return inliers;
}

vector<Moments> Shape::getMoments(vector<vector<Point> > contours){

	//###############################################################################
	// FUNCTION: 
	// Using cv::moments to compute image moments and store them as <vector<Moments>>
	// (see OpenCV documentation).
	//
	// INPUT: 
	// vector<vector<Point> > contours: list of contour pixels
	//
	// OUTPUT:
	// <vector<Moments>> mom
	//###############################################################################

	//*******************************************************************************
	// Variables
	vector<Moments> mom(contours.size());

	//*******************************************************************************
	// Computing each moment for each shape
	for (size_t i = 0; i < contours.size(); i++){
		mom[i] = moments(contours[i], false);
	}

	return mom;
}

vector<double> Shape::areaSize(vector<vector<Point> > contours, vector<Moments> moment){

	//###############################################################################
	// FUNCTION: Extracts for each shape the moment of first order, which equals the
	// area size of the shape.
	//
	// INPUT: 
	// vector<vector<Point> > contours: list of contour pixels
	// <vector<Moments>> moments: image moments of all shapes
	//
	// OUTPUT:
	// vector<double> area: area size (number of pixels) for each ridge/shape
	//###############################################################################

	//*******************************************************************************
	// Variables
	vector<double> area(contours.size());

	//*******************************************************************************
	// Extract area size for each shape/ridge from image moment
	for (size_t i = 0; i < contours.size(); i++){
		area[i] = moment[i].m00;
	}

	return area;
}

vector<double> Shape::contourLength(vector<vector<Point> > contours){
	
	//###############################################################################
	// FUNCTION: Computes the length of the contour for each shape/ridge using the
	// cv::arcLength function (see OpenCV documentation). 
	//
	// INPUT: 
	// vector<vector<Point> > contours: list of contour pixels
	//
	// OUTPUT:
	// vector<double> Length: contour Length for each ridge/shape
	//###############################################################################

	//*******************************************************************************
	// Variables
	vector<double> Length;

	//*******************************************************************************
	// Computes the length of the contour for each shape/ridge 
	for (size_t i = 0; i < contours.size(); i++){

		Length.push_back(arcLength(contours[i], true));
	}

	return Length;
}

vector<Point2d> Shape::centMass(vector<vector<Point> > contours, vector<Moments> moment){

	//###############################################################################
	// FUNCTION: 
	// 
	//
	// INPUT: 
	// 
	//
	// OUTPUT:
	// 
	//###############################################################################

	//*******************************************************************************
	// Variables
	vector<Point2d> massCenter(contours.size());

	//*******************************************************************************
	// 
	for (size_t i = 0; i < contours.size(); i++){
		massCenter[i] = Point2d(moment[i].m10 / moment[i].m00, moment[i].m01 / moment[i].m00);
	}

	return massCenter;
}


vector<Point2d> Shape::shapeAxis(vector<vector<Point> > contours){
	//###############################################################################
	// FUNCTION: 
	// 
	//
	// INPUT: 
	// 
	//
	// OUTPUT:
	// 
	//###############################################################################

	//*******************************************************************************
	// Variables
	vector<Point2d> axis;
	double distance;
	double max;
	double x_1, x_2, y_1, y_2;
	double m, angle;

	//*******************************************************************************
	// 
	for (size_t i = 0; i < contours.size(); i++){
	max = 0;
	x_1 = 0;
	x_2 = 0;
	y_1 = 0;
	y_2 = 0;
	m = 0;

		for (size_t j = 0; j < contours[i].size(); j++){
			for (size_t k = 0; k < contours[i].size(); k++){

				distance = sqrt(pow((contours[i][j].x - contours[i][k].x), 2) + pow(contours[i][j].y - contours[i][k].y, 2));
				double x1 = contours[i][j].x;
				double x2 = contours[i][k].x;
				double y1 = contours[i][j].y;
				double y2 = contours[i][k].y;

				if (distance > max) {
					max = distance;
					x_1 = x1;
					x_2 = x2;
					y_1 = y1;
					y_2 = y2;
				}
			}
		}
		m = (y_2 - y_1) / (x_2 - x_1);
		if (m < 0){
			angle = atan(abs(m)) / M_PI * 180 + 90;
		}
		else{
			angle = atan(abs(m)) / M_PI * 180;
		}
		
		axis.push_back(Point2d(max, angle));

	}


	return axis;
}

MatrixXd Shape::convertVector(vector<Point2d> massCenter, vector<double> Lengths, vector<double> Areas, vector<Point2d> Axis){

	//###############################################################################
	// FUNCTION: 
	// 
	//
	// INPUT: 
	// 
	//
	// OUTPUT:
	// 
	//###############################################################################

	//*******************************************************************************
	// Variables
	vector<Point2d> centroids = massCenter;
	vector<double> length = Lengths;
	vector<double> area = Areas;
	vector<Point2d> axes = Axis;
	MatrixXd Properties(8, centroids.size());

	//*******************************************************************************
	//
	for (size_t i = 0; i < centroids.size(); i++){
		Properties(0, i) = area[i];
		Properties(1, i) = length[i];
		Properties(2, i) = axes[i].x;
		Properties(3, i) = axes[i].y;
		Properties(4, i) = 0;
		Properties(5, i) = centroids[i].x;
		Properties(6, i) = centroids[i].y;
		Properties(7, i) = 0;
	}

	return Properties;
}

MatrixXd Shape::sortMatrix(MatrixXd Properties){

	//###############################################################################
	// FUNCTION: 
	// 
	//
	// INPUT: 
	// 
	//
	// OUTPUT:
	// 
	//###############################################################################

	//*******************************************************************************
	// Variable
	MatrixXd Prop = Properties;
	MatrixXd sortedProperties(8, Prop.cols());
	MatrixXd features;
	double areaThresh = 40;
	int ThreshIndex;

	//*******************************************************************************
	//
	for (int i = 0; i < Prop.cols(); i++){

		MatrixXd::Index minRow;
		double min = Prop.row(0).minCoeff(&minRow);
		sortedProperties.col(i) = Prop.col(minRow);
		Prop(0, minRow) = 100000;
	}

	//*******************************************************************************
	//
	for (int i = 0; i < sortedProperties.cols(); i++){
		if (sortedProperties(0, i) >= areaThresh){
			ThreshIndex = i;

			goto SELECT;
		}
	}

	//*******************************************************************************
	//
	SELECT:
	features = sortedProperties.block(0, ThreshIndex, 8, sortedProperties.cols() - ThreshIndex);

	return features; //sortedProperties;
}

double Shape::mainAxis(MatrixXd features){

	//###############################################################################
	// FUNCTION: 
	// 
	//
	// INPUT: 
	// 
	//
	// OUTPUT:
	// 
	//###############################################################################

	//*******************************************************************************
	// Variable
	double weightedSum = 0;
	double lineSum = 0;
	double mainAx;
	int consensus = 0;
	double delta_angle = 30;
	int temp_cons;
	int max_index = 0;

	//*******************************************************************************
	//Pseudo-RANSAC
	for (int i = 0; i < features.cols(); i++){
		temp_cons = 0;

		for (int j = 0; j < features.cols(); j++){
			if ((features(3, i) - delta_angle <= features(3, j)) && (features(3, j) >= features(3, i) + delta_angle)){
				temp_cons++;
			}
		}

		if (temp_cons > consensus){
			consensus = temp_cons;
			max_index = i;
		}
	}
	//*******************************************************************************
	//
	for (int i = 0; i < features.cols(); i++){
		
		if ((features(3, max_index) - delta_angle <= features(3, i)) && (features(3, i) >= features(3, max_index) + delta_angle)){
			lineSum = lineSum + features(2, i);
			weightedSum = weightedSum + features(2, i)*features(3, i);
		}

	}
	mainAx = weightedSum / lineSum;

	return  mainAx;
}

MatrixXd Shape::axisDiff(MatrixXd featuresSorted, double mainAxis){
	
	//###############################################################################
	// FUNCTION: 
	// 
	//
	// INPUT: 
	// 
	//
	// OUTPUT:
	// 
	//###############################################################################

	//*******************************************************************************
	// 
	MatrixXd features = featuresSorted;

	//*******************************************************************************
	//
	for (int i = 0; i < features.cols(); i++){
		features(4, i) = mainAxis - features(3, i);
	}
	
	return features;
}

int Shape::matchFeatures(MatrixXd movingMat, MatrixXd staticMat){

	//###############################################################################
	// FUNCTION: 
	// 
	//
	// INPUT: 
	// 
	//
	// OUTPUT:
	// 
	//###############################################################################

	//*******************************************************************************
	// 

	int count = 0;
	double delta_area = 0.1;
	double delta_contour = 0.2;
	double delta_line = 5;
	double delta_angle = 20;
	double radius = 60;

	//*******************************************************************************
	// 	
	for (int i = 0; i < staticMat.cols(); i++){
	for (int j = 0; j < movingMat.cols(); j++){

		if ((movingMat(2, j) >= staticMat(2, i) - delta_line) && (movingMat(2, j) <= staticMat(2, i) + delta_line)
			&& (movingMat(0, j) >= staticMat(0, i)* (1 - delta_area)) && (movingMat(0, j) <= staticMat(0, i)*(1 + delta_area))
			&& (movingMat(1, j) >= staticMat(1, i)*(1- delta_contour)) && (movingMat(1, j) <= staticMat(1, i) * (1+delta_contour))
			&& (movingMat(4, j) >= staticMat(4, i) - delta_angle)       && (movingMat(4, j) <= staticMat(4, i) + delta_angle)
			&& (sqrt(pow(movingMat(5, j) - staticMat(5, i), 2) + pow(movingMat(6, j) - staticMat(6, i), 2)) <= radius)
			){
				count++;
			}
		}
	}

	return count;
}

MatrixXd Shape::computeNeighbor(MatrixXd Mat, vector<int> index){

	//###############################################################################
	// FUNCTION: 
	// 
	//
	// INPUT: 
	// 
	//
	// OUTPUT:
	// 
	//###############################################################################

	//*******************************************************************************
	// Variables
	MatrixXd distances(index.size(),3);
	MatrixXd distancesSorted(index.size(), 3);
	MatrixXd neighbor(index.size(), 10);
	vector<vector<double>> distancesMov;
	MatrixXd::Index minCol;

	//*******************************************************************************
	//
	for (size_t i = 0; i < index.size(); i++){
 
		for (size_t j = 0; j, j < index.size(); j++){
			distances(j, 0) = index[j];
			distances(j, 1) = sqrt(pow(Mat(5, index[i]) - Mat(5, index[j]), 2) + pow(Mat(6, index[i]) - Mat(6, index[j]), 2));
			distances(j, 2) = Mat(3, index[i]) - Mat(3, index[j]);
		}

		for (int k = 0; k < distances.rows(); k++){
			double min = distances.col(1).minCoeff(&minCol);
			distancesSorted.row(k) = distances.row(minCol);
			distances(minCol, 1) = 100000;
		}

		neighbor(i, 0) = index[i];

		unsigned int l = 0;
		unsigned int c = 0;
		do{
			if (distancesSorted(l + 1, 0) != distancesSorted(l, 0)){
				neighbor.block(i, (1 + 3 * c), 1, 3) = distancesSorted.row(l + 1);
				c++;
			}
			l++;
		} while (c < 3);
	}

	return neighbor;
}

vector<Point2i> Shape::compareNeighbor(MatrixXd neighborStat, MatrixXd neighborMov, vector<int> index_stat, vector<int> index_mov, vector<Point2i> index){
	
	//###############################################################################
	// FUNCTION: 
	// 
	//
	// INPUT: 
	// 
	//
	// OUTPUT:
	// 
	//###############################################################################

	//*******************************************************************************
	// Variables
	vector<Point2i> assignment;
	double threshold = 1.5;
	double threshold_distance = 0.05;
	double threshold_angle = 0.1;
	bool neighbor = false;
	int count = 0;

	//*******************************************************************************
	// 
	for (size_t i = 0; i < index.size(); i++){
		count = 0;
		for (size_t j = 0; j < index.size(); j++){


				if (Point2i(neighborStat(i, 1), neighborMov(i, 1)) == index[j]){
					count++;
				}

				if (Point2i(neighborStat(i, 4), neighborMov(i, 4)) == index[j]){
					count++;
				}

				if (Point2i(neighborStat(i, 7), neighborMov(i, 7)) == index[j]){
					count++;
				}
				
				if (count == 3){
					assignment.push_back(Point2i(neighborStat(i, 0), neighborMov(i, 0)));
					break;
				}
				
		}

	}

	return assignment;
}

vector<Point2i> Shape::pickPoints(MatrixXd staticMat, MatrixXd movingMat){
	
	//###############################################################################
	// FUNCTION: 
	// 
	//
	// INPUT: 
	// 
	//
	// OUTPUT:
	// 
	//###############################################################################

	//*******************************************************************************
	// Variables
	Shape S;
	double delta_area = 0.1;
	double delta_contour = 0.2;
	double delta_line = 2.5;
	double delta_angle = 20;
	double radius = 60;
	vector<Point2i> index, assigned;
	vector<int> index_stat, index_mov;
	MatrixXd neighborStatic, neighborMoving;

	//*******************************************************************************
	// 	
	for (int i = 0; i < staticMat.cols(); i++){
		for (int j = 0; j < movingMat.cols(); j++){
			if ((movingMat(2, j) >= staticMat(2, i) - delta_line) && (movingMat(2, j) <= staticMat(2, i) + delta_line)
				&& (movingMat(0, j) >= staticMat(0, i)* (1 - delta_area)) && (movingMat(0, j) <= staticMat(0, i)*(1 + delta_area))
				&& (movingMat(1, j) >= staticMat(1, i)*(1 - delta_contour)) && (movingMat(1, j) <= staticMat(1, i) * (1 + delta_contour))
				//&& (movingMat(4, j) >= staticMat(4, i) - delta_angle) && (movingMat(4, j) <= staticMat(4, i) + delta_angle)
				&& (sqrt(pow(movingMat(5, j) - staticMat(5, i), 2) + pow(movingMat(6, j) - staticMat(6, i), 2)) <= radius) 
				){
				index.push_back(Point(i, j));
				index_stat.push_back(i);
				index_mov.push_back(j);
			}
		}
	}

	//*******************************************************************************
	// 	
	neighborStatic = S.computeNeighbor(staticMat, index_stat);
	neighborMoving = S.computeNeighbor(movingMat, index_mov);

	//*******************************************************************************
	// 	
	assigned = S.compareNeighbor(neighborStatic, neighborMoving, index_stat, index_mov, index);

	return assigned;
}

MatrixXd Shape::extractFeatures(Mat input, int roiSize){

	//###############################################################################
	// FUNCTION: 
	// 
	//
	// INPUT: 
	// 
	//
	// OUTPUT:
	// 
	//###############################################################################

	//*******************************************************************************
	// Variable
	Shape S;
	MatrixXd PropSorted;
	vector<vector<Point> > Contours;
	vector<Moments> Moments;
	vector<Point2d> MassCenter;
	vector<double> Areas;
	vector<double> Lengths;
	vector<Point2d> Axis;
	MatrixXd Properties;
	MatrixXd features;
	double mainAxis;

	//*******************************************************************************
	// Extract contours of ridge areas, incomplete ridge areas are filtered out.
	Contours = S.findCont(input, roiSize);

	//*******************************************************************************
	// Compute image moments for extracted contours.
	Moments = S.getMoments(Contours);

	//*******************************************************************************
	// Calculate centroids for ridge areas.
	MassCenter = S.centMass(Contours, Moments);

	//*******************************************************************************
	// Calculate area size for each ridge area.
	Areas = S.areaSize(Contours, Moments);

	//*******************************************************************************
	// Calculate contour length for each ridge area.
	Lengths = S.contourLength(Contours);

	//*******************************************************************************
	//
	Axis = S.shapeAxis(Contours);

	//*******************************************************************************
	// Summarize Area, Length and (x,y,z) coordinates of centroid in one matrix for
	// all shapes. Matrix has dimensions (5 x M), each column represents one shape.
	Properties = S.convertVector(MassCenter, Lengths, Areas, Axis);

	//*******************************************************************************
	// Matrix with properties of each shapes is sorted by the size of the areas.
	PropSorted = S.sortMatrix(Properties);
	
	//*******************************************************************************
	// Matrix with properties of each shapes is sorted by the size of the areas.
	mainAxis = S.mainAxis(PropSorted);

	//*******************************************************************************
	// 
	PropSorted = axisDiff(PropSorted, mainAxis);

	return PropSorted;
}

vector<Point3i> Shape::firstSearch(Mat src, int roiSize, MatrixXd staticM){

	//###############################################################################
	// FUNCTION: 
	// 
	//
	// INPUT: 
	// 
	//
	// OUTPUT:
	// 
	//###############################################################################

	//*******************************************************************************
	// Variables
	Shape S;
	int step = roiSize/2;
	int rest_x = src.cols % step;
	int rest_y = src.rows % step;
	int number_x = ((src.cols - rest_x) / step) - 1 ;
	int number_y = ((src.rows - rest_y) / step) - 1;
	int matchCounter = 0;
	vector<Point3i> counts;

	//*******************************************************************************
	//
	for (int i = 0; i < number_x; i++){
		for (int j = 0; j < number_y; j++){
			movingROI = S.setRoi(src, roiSize, i*step, j*step);
			movingPropSorted = S.extractFeatures(movingROI, roiSize);
			matchCounter = S.matchFeatures(movingPropSorted, staticM);
			counts.push_back(Point3i(i*step, j*step, matchCounter));
		}

	}

	//*******************************************************************************
	//
	if (rest_x != 0 && rest_y == 0){
		for (int j = 0; j < number_y; j++){
			int i = src.cols - roiSize;
			movingROI = S.setRoi(src, roiSize, i, j*step);
			movingPropSorted = S.extractFeatures(movingROI, roiSize);
			matchCounter = S.matchFeatures(movingPropSorted, staticM);
			counts.push_back(Point3i(i, j*step, matchCounter));
		}
	}

	//*******************************************************************************
	//
	if (rest_y != 0 && rest_x ==0){
		for (int i = 0; i < number_x; i++){
			int j = src.rows - roiSize;
			movingROI = S.setRoi(src, roiSize, i*step, j);
			movingPropSorted = S.extractFeatures(movingROI, roiSize);
			matchCounter = S.matchFeatures(movingPropSorted, staticM);
			counts.push_back(Point3i(i*step, j, matchCounter));
		}
	}
	
	//*******************************************************************************
	//
	if (rest_y != 0 && rest_x != 0){
		for (int j = 0; j < number_y; j++){
			int i = src.cols - roiSize;
			movingROI = S.setRoi(src, roiSize, i, j*step);
			movingPropSorted = S.extractFeatures(movingROI, roiSize);
			matchCounter = S.matchFeatures(movingPropSorted, staticM);
			counts.push_back(Point3i(i, j*step, matchCounter));
		}

		for (int i = 0; i < number_x; i++){
			int j = src.rows - roiSize;
			movingROI = S.setRoi(src, roiSize, i*step, j);
			movingPropSorted = S.extractFeatures(movingROI, roiSize);
			matchCounter = S.matchFeatures(movingPropSorted, staticM);
			counts.push_back(Point3i(i*step, j, matchCounter));
		}
		
		//*******************************************************************************
		//
		int i = src.cols - roiSize;
		int j = src.rows - roiSize;

		movingROI = S.setRoi(src, roiSize, i, j);
		movingPropSorted = S.extractFeatures(movingROI, roiSize);
		matchCounter = S.matchFeatures(movingPropSorted, staticM);
		counts.push_back(Point3i(i, j, matchCounter));

	}

	return counts;
}

Point3i Shape::fineSearch(Mat src, Point3i maxMatch, int roiSize, MatrixXd staticM){

	//###############################################################################
	// FUNCTION: 
	// 
	//
	// INPUT: 
	// 
	//
	// OUTPUT:
	// 
	//###############################################################################

	//*******************************************************************************
	// 
	int xLeft = maxMatch.x;
	int xRight = maxMatch.x;
	int yUp = maxMatch.y;
	int yDown = maxMatch.y;
	int newX = xLeft;
	int newY = yUp;;
	int upcount = 0;
	int downcount = 0;
	int leftcount = 0;
	int rightcount = 0;
	int matchCounter;
	int counts = maxMatch.z;

	//*******************************************************************************
	//
	if (yUp != 0){
		do{
			yUp = yUp - 10;
			movingROI = setRoi(src, roiSize, xLeft, yUp);
			movingPropSorted = extractFeatures(movingROI, roiSize);
			matchCounter = matchFeatures(movingPropSorted, staticM);

			if (matchCounter > counts){
				counts = matchCounter;
				newY = yUp;
			}

			upcount++;

		} while (upcount < 5 && yUp != 0);
	}

	//*******************************************************************************
	//
	if (yDown != src.rows - roiSize){
		do{
			yDown = yDown + 10;
			movingROI = setRoi(src, roiSize, xLeft, yDown);
			movingPropSorted = extractFeatures(movingROI, roiSize);
			matchCounter = matchFeatures(movingPropSorted, staticM);

			if (matchCounter > counts){
				counts = matchCounter;
				newY = yDown;
			}

			downcount++;

		} while (downcount < 5 && yDown <= src.rows - roiSize);
	}

	//*******************************************************************************
	//
	if (xLeft != 0){
		do
		{
			xLeft = xLeft - 10;
			movingROI = setRoi(src, roiSize, xLeft, newY);
			movingPropSorted = extractFeatures(movingROI, roiSize);
			matchCounter = matchFeatures(movingPropSorted, staticM);

			if (matchCounter > counts){
				counts = matchCounter;
				newX = xLeft;
			}

			leftcount++;

		} while (leftcount < 5 && xLeft !=0);
	}

	//*******************************************************************************
	//
	if (xRight != src.cols - roiSize){
		do
		{
			xRight = xRight + 10;
			movingROI = setRoi(src, roiSize, xLeft, newY);
			movingPropSorted = extractFeatures(movingROI, roiSize);
			matchCounter = matchFeatures(movingPropSorted, staticM);

			if (matchCounter > counts){
				counts = matchCounter;
				newX = xRight;
			}

			rightcount++;

		} while (leftcount < 5 && xRight <= src.cols - roiSize);
	}
	return Point3i(newX, newY, counts);
}

void Shape::location(Mat input, Mat ROI, int roiSize, float scale){

	//###############################################################################
	// FUNCTION: 
	// 
	//
	// INPUT: 
	// 
	//
	// OUTPUT:
	// 
	//###############################################################################

	//*******************************************************************************
	// Variable
	Shape S;
	Vector3f corner;
	vector<Point3i> matchCount;
	Point3i coordinates;
	int maxIndex;
	int max = 0;
	int type = 1;
	int row = input.rows;
	int col = input.cols;
	double scale_ = 1;
	vector<Point2i> indices;
	MatrixXd r_red, l_red;
	Matrix3d Rotation;
	Vector3d translation, coord, t_new;

	//###############################################################################
	// Extract Features for static ROI (patch)
	staticROI = ROI;
	staticPropSorted = S.extractFeatures(staticROI, roiSize);
	staticMainAxis = mainAxis(staticPropSorted);

	//cout << staticPropSorted.cols() << endl;
	
	//###############################################################################
	// 
	matchCount = firstSearch(input, roiSize, staticPropSorted);

	for (size_t i = 0; i < matchCount.size(); i++)
	{
		if (matchCount[i].z > max){
			max = matchCount[i].z;
			maxIndex = i;
		}
	}

	//###############################################################################
	// 
	coordinates = fineSearch(input, matchCount[maxIndex], roiSize, staticPropSorted);
	coord << coordinates.x, coordinates.y, 0;
	newROI = setRoi(input, roiSize, coordinates.x, coordinates.y);
	newPropSorted = S.extractFeatures(newROI, roiSize);
	newMainAxis = mainAxis(newPropSorted);

	//###############################################################################
	// Pick suitable point pairs for parameter estimation
	indices = S.pickPoints(staticPropSorted, newPropSorted);

	//###############################################################################
	MatrixXd left(3, indices.size());
	MatrixXd right(3, indices.size());

	for (size_t i = 0; i < indices.size(); i++){
		left(0, i) = staticPropSorted(5, indices[i].x);
		left(1, i) = staticPropSorted(6, indices[i].x);
		left(2, i) = 0;

		right(0, i) = newPropSorted(5, indices[i].y);
		right(1, i) = newPropSorted(6, indices[i].y);
		right(2, i) = 0;
	}

	//cout << "left" << endl << left << endl;
	//cout << "Right" << endl << right << endl;

	//cout << left.cols() << endl;

	//*******************************************************************************
	// Perform parameter estimation for Rotation and translation between point sets

	r_red = reducePoints(right);
	l_red = reducePoints(left);

	Rotation = S.computeR(l_red, r_red);

//	cout << "Rotation" << endl << Rotation << endl;

	if (Rotation(0, 1) < 0){
		
		translation = S.computeTranslation(right, left, scale_, Rotation);
	
	} else{
		translation = S.computeTranslation(right, left, scale_, Rotation.transpose());
		//cout << "Translation" << endl << translation << endl;
	}

	t_new = Rotation.transpose()*coord + translation - Rotation*coord;

	//*******************************************************************************
	// Mark patch in image
	imshow("ROI", staticROI);
	S.drawROI(input, coordinates, Rotation, t_new, scale_, roiSize);

}

void Shape::drawROI(Mat input, Point3i coordinates, Matrix3d Rotation, Vector3d translation, double scale, int roiSize){
	
	//###############################################################################
	// FUNCTION: 
	// 
	//
	// INPUT: 
	// 
	//
	// OUTPUT:
	// 
	//###############################################################################

	//*******************************************************************************
	//
	Mat drawROI;	
	MatrixXd corners(3, 4);
	MatrixXd cornersTrans(3, 4);

	//*******************************************************************************
	//
	cvtColor(input, drawROI, cv::COLOR_GRAY2RGB);

	//*******************************************************************************
	//
	corners << coordinates.x, coordinates.x + roiSize, coordinates.x, coordinates.x + roiSize,
			   coordinates.y, coordinates.y, coordinates.y + roiSize, coordinates.y + roiSize,
			    0, 0, 0, 0;
	
	//*******************************************************************************
	//
	cornersTrans = transformMat(corners, Rotation, translation, scale);

	//*******************************************************************************
	//
	line(drawROI, Point(cornersTrans(0, 0), cornersTrans(1, 0)), Point(cornersTrans(0, 1), cornersTrans(1, 1)), Scalar(0, 225, 0), 2);
	line(drawROI, Point(cornersTrans(0, 2), cornersTrans(1, 2)), Point(cornersTrans(0, 3), cornersTrans(1, 3)), Scalar(0, 225, 0), 2);
	line(drawROI, Point(cornersTrans(0, 0), cornersTrans(1, 0)), Point(cornersTrans(0, 2), cornersTrans(1, 2)), Scalar(0, 225, 0), 2);
	line(drawROI, Point(cornersTrans(0, 1), cornersTrans(1, 1)), Point(cornersTrans(0, 3), cornersTrans(1, 3)), Scalar(0, 225, 0), 2);

	//*******************************************************************************
	//
	imshow("Draw", drawROI);
}

int main(int argc, char** argv)

{
	//###############################################################################
	// FUNCTION: 
	// 
	//
	// INPUT: 
	// 
	//
	// OUTPUT:
	// 
	//###############################################################################

	//*******************************************************************************
	//
	Shape S;
	clock_t t;
	t = clock();
	srand((time(NULL)));

	//*******************************************************************************
	// Variables
	int roiSize;
	int scale = 1;
	double epsilon = 1.0e-020;

	//*******************************************************************************
	//  Read in image
	Mat src = imread("src_small_2.png", 0);
	//Mat ROI = imread("ROI_250.png", 0);
	Mat ROI = imread("patch_shear_-5.png", 0);
	roiSize = ROI.rows;

	//*******************************************************************************
	// 
	S.location(src, ROI, roiSize, scale);


	//*******************************************************************************
	// 
	t = clock() - t;
	std::cout << "time: " << t*1.0 / CLOCKS_PER_SEC << " seconds" << endl;

	waitKey(0);
	return 0;
}