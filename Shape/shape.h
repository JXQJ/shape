#ifndef SHAPE_H
#define SHAPE_H

//----- Includes -----
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
#include <sstream>
#include <iostream>

#include <Eigen/Dense>
#include <Eigen/Core>
#include <EIgen/Eigenvalues>

using namespace cv;
using namespace std;
using namespace Eigen;

class Shape
{


public:

	//###############################################################################
	// VARIABLES
	//###############################################################################

	// --- General --- 
	int RoiSize;
	
	//*******************************************************************************
	// Variables static ROI
	Mat staticROI;
	MatrixXd staticPropSorted;
	double staticMainAxis;

	//*******************************************************************************
	// Variables moving ROI
	Mat movingROI;
	MatrixXd movingPropSorted;

	// Final ROI
	Mat newROI;
	MatrixXd newPropSorted;
	double newMainAxis;

	//###############################################################################
	// FUNCTIONS
	//###############################################################################
	
	//*******************************************************************************
	// Main and Secondary function
	Mat main(int, char** argv);
	void location(Mat input, Mat ROI, int roiSize, float scale);

	//*******************************************************************************
	// Setting ROI for image
	Mat setRoi(Mat input, int roiSize, int x, int y);

	//*******************************************************************************
	//  Finding features
	vector<vector<Point>> findCont(Mat input, int roiSize);
	vector<Moments> getMoments(vector<vector<Point> > contours);
	vector<double> areaSize(vector<vector<Point> > contours, vector<Moments> moment);
	vector<double> contourLength(vector<vector<Point> > contours);
	vector<Point2d> centMass(vector<vector<Point> > contours, vector<Moments> moment);
	vector<Point2d> shapeAxis(vector<vector<Point> > contours);
	double mainAxis(MatrixXd features);
	MatrixXd axisDiff(MatrixXd Properties, double mainAxis);

	//*******************************************************************************
	// Matching features
	MatrixXd extractFeatures(Mat input, int roiSize);
	int matchFeatures(MatrixXd movingMat, MatrixXd staticMat);

	//*******************************************************************************
	// Helper functions
	MatrixXd convertVector(vector<Point2d> massCenter, vector<double> Lengths, vector<double> Areas, vector<Point2d> Axis);
	MatrixXd sortMatrix(MatrixXd Properties);
	
	
	//*******************************************************************************
	// Find Global location
	vector<Point3i> firstSearch(Mat src, int roiSize, MatrixXd staticM);
	Point3i fineSearch(Mat input, Point3i maxMatch, int roiSize, MatrixXd staticM);
	MatrixXd transformMat(MatrixXd input, Matrix3d Rotation, VectorXd translation, double scale);

	//*******************************************************************************
	// Pick points
	vector<Point2i> pickPoints(MatrixXd staticM, MatrixXd movingM);
	MatrixXd computeNeighbor(MatrixXd Mat, vector<int> index);
	vector<Point2i> compareNeighbor(MatrixXd neighborStat, MatrixXd neighborMov, vector<int> index_stat, vector<int> index_mov, vector<Point2i> index);
	
	//*******************************************************************************
	// Parameter Estimation
	double computeScale(MatrixXd left, MatrixXd right);
	MatrixXd reducePoints(MatrixXd pointcloud);
	Matrix4d computeS(MatrixXd left_red, MatrixXd right_red);
	Matrix3d computeR(MatrixXd left, MatrixXd right);
	Vector3d computeTranslation(MatrixXd left, MatrixXd right, double scale, Matrix3d Rotation);

	
	//*******************************************************************************
	//  Draw
	void drawROI(Mat input, Point3i corner, Matrix3d Rotation, Vector3d translation, double scale, int roiSize);

	//*******************************************************************************
	// ICP
	// MatrixXf assignPoints(MatrixXf leftTransformed, MatrixXf right);
	// Vector4f ICP(MatrixXf left, MatrixXf right, float epsilon);
	// Vector3f transformPoint(Vector3f input, Matrix3f Rotation, Vector3f translation, float scale);
};

#endif // SHAPE_H