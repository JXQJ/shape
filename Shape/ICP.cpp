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

#include "shape.h"

#include <Eigen/Dense>
#include <Eigen/Core>
#include <EIgen/Eigenvalues>

using namespace cv;
using namespace std;
using namespace Eigen;

MatrixXd Shape::transformMat(MatrixXd input, Matrix3d Rotation, VectorXd translation, double scale){

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
	MatrixXd Rotated(3, input.cols());
	MatrixXd Transformed(3, input.cols());

	Rotated = Rotation*input;
	Rotated *= scale;

	for (int i = 0; i < input.cols(); i++){
		Transformed.col(i) = Rotated.col(i) + translation;
	}

	return Transformed;
}

Matrix4d Shape::computeS(MatrixXd left_red, MatrixXd right_red){

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
	Matrix4d S;
	double S_xx, S_yy, S_zz, S_xy, S_xz, S_yx, S_yz, S_zx, S_zy;

	S_xx = left_red.row(0).dot(right_red.row(0));
	S_yy = left_red.row(1).dot(right_red.row(1));
	S_zz = left_red.row(2).dot(right_red.row(2));
	S_xy = left_red.row(0).dot(right_red.row(1));
	S_xz = left_red.row(0).dot(right_red.row(2));
	S_yx = left_red.row(1).dot(right_red.row(0));
	S_yz = left_red.row(1).dot(right_red.row(2));
	S_zx = left_red.row(2).dot(right_red.row(0));
	S_zy = left_red.row(2).dot(right_red.row(1));

	S << S_xx + S_yy + S_zz,	S_yz - S_zy,			S_zx - S_xz,			S_xy - S_yx, 
		 S_yz - S_zy,			S_xx - S_yy - S_zz,		S_xy + S_yx,			S_zx + S_xz, 
		 S_zx - S_xz,			S_xy + S_yx,			-S_xx + S_yy - S_zz,	S_yz + S_zy,
		 S_xy - S_yx,			S_zx + S_xz,			S_yz + S_zy,			-S_xx - S_yy + S_zz;

	//cout << S << endl; 
	return S;
}

Matrix3d Shape::computeR(MatrixXd left, MatrixXd right){

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
	Matrix4d SM = computeS(left, right);

	EigenSolver<Matrix4d> es(SM);
	VectorXd eigenVal(es.eigenvalues().size());


	for (int i = 0; i < es.eigenvalues().size(); i++){
		eigenVal(i) = es.eigenvalues()[i].real();
	}

	//cout << eigenVal << endl;
	//cout << es.eigenvectors() << endl;

	std::ptrdiff_t i;
	double max = eigenVal.maxCoeff(&i);

	VectorXcd eigenVec(es.eigenvalues().size());
	eigenVec = (es.eigenvectors().col(i));

	Vector4d q(4);
	q(0) = eigenVec(0).real();
	q(1) = eigenVec(1).real();
	q(2) = eigenVec(2).real();
	q(3) = eigenVec(3).real();

	//cout << q << endl;
	Matrix3d R;

	R << (q(0)*q(0) + q(1)*q(1) - q(2)*q(2) - q(3)*q(3)) / (q(0)*q(0) + q(1)*q(1) + q(2)*q(2) + q(3)*q(3)), 
		2 * (q(1)*q(2) - q(0)*q(3)) / (q(0)*q(0) + q(1)*q(1) + q(2)*q(2) + q(3)*q(3)), 
		2 * (q(1)*q(3) + q(0)*q(2)) / (q(0)*q(0) + q(1)*q(1) + q(2)*q(2) + q(3)*q(3)),
		2 * (q(2)*q(1) + q(0)*q(3)) / (q(0)*q(0) + q(1)*q(1) + q(2)*q(2) +  q(3)*q(3)), 
		(q(0)*q(0) - q(1)*q(1) + q(2)*q(2) - q(3)*q(3)) / (q(0)*q(0) + q(1)*q(1) + q(2)*q(2) + q(3)*q(3)), 
		2 * (q(2)*q(3) - q(0)*q(1)) / (q(0)*q(0) + q(1)*q(1) + q(2)*q(2) + q(3)*q(3)),
		2 * (q(3)*q(1) - q(0)*q(2)) / (q(0)*q(0) + q(1)*q(1) + q(2)*q(2) + q(3)*q(3)),
		2 * (q(3)*q(2) + q(0)*q(1)) / (q(0)*q(0) + q(1)*q(1) + q(2)*q(2) + q(3)*q(3)), 
		(q(0)*q(0) - q(1)*q(1) - q(2)*q(2) + q(3)*q(3)) / (q(0)*q(0) + q(1)*q(1) + q(2)*q(2) + q(3)*q(3));

	return R;
}

Vector3d Shape::computeTranslation(MatrixXd left, MatrixXd right, double scale, Matrix3d Rotation){

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
	Vector3d t, r_, l_, rightTrans;

	for (int i = 0; i < 3; i++){
		l_(i) = (left.row(i).sum() / left.cols());
		r_(i) = (right.row(i).sum() / right.cols());
	}

	rightTrans = Rotation*r_;
	rightTrans *= scale;
	t = l_ - rightTrans;

	return t;
}

double Shape::computeScale(MatrixXd left, MatrixXd right){

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
	double scale;
	scale = right.norm() / left.norm();
	return scale;
}

MatrixXd Shape::reducePoints(MatrixXd pointcloud){

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
	MatrixXd  p_red(pointcloud.rows(), pointcloud.cols());
	Vector3d p_;
	
	for (int i = 0; i < 3; i++){
	p_(i) = (pointcloud.row(i).sum()/pointcloud.cols());
	}

	for (int i = 0; i < pointcloud.cols(); i++){
		p_red.col(i) = pointcloud.col(i) - p_;
	}

	return p_red;
}

/*Vector4f Shape::ICP(MatrixXf left, MatrixXf right, float epsilon){

	double alpha = 355*M_PI/180;

	Matrix3f R_init;
	R_init << cos(alpha), -sin(alpha), 0,
		sin(alpha), cos(alpha), 0,
		0, 0, 1; 

	Vector3f t_init;
	t_init << 10, 10, 0;

	float scale_init = 1;

	MatrixXf l_transformed = transformMat(left, R_init, t_init, scale_init);

	int c = 0;
	float mse = 0;
	float mseDiff = 10000000000000000;
	
	Matrix3f R_e;
	Vector3f t_e;
	float scale_e;
	
	while (mseDiff > epsilon){
		
		c = c + 1;
		float mse_2 = mse;
		
		MatrixXf r_new = assignPoints(l_transformed, right);
		
		MatrixXf r_red = reducePoints(r_new);
		MatrixXf l_red = reducePoints(left);

		cout << r_red << endl << l_red << endl;

		R_e = computeR(l_red , r_red);
		scale_e = computeScale(l_red, r_red);
		t_e = computeTranslation(left, r_new, scale_e, R_e);
		
		l_transformed = transformMat(left, R_e, t_e, scale_e);

		mse = pow(((right - l_transformed).sum()), 2);
		mseDiff = abs(mse - mse_2);
	}

	/*cout << "R" << endl << R_e << endl;
	cout << "t" << endl << t_e << endl;
	cout << "count" << endl << c << endl;

	Vector4f parameters;
	parameters << acos(R_e(0, 0)), t_e(0), t_e(1), scale_e;
	

	return parameters;
}*/

/*MatrixXf Shape::assignPoints(MatrixXf left, MatrixXf right){
	
	VectorXf distance(left.cols());
	MatrixXf r_new = right; //(3, left.cols());
	MatrixXf r = right;
	Vector3f inf;
	inf << 10000000, 10000000, 10000000;

	for (int i = 0; i < left.cols(); i++){
		
		for (int j = 0; j < left.cols(); j++){

			distance(j) = sqrt(pow(r(0, j) - left(0, i), 2) + pow(r(1, j) - left(1, i), 2) );
		}

		VectorXf::Index minRow;
		float min = distance.minCoeff(&minRow);
		r_new.col(i) = r.col(minRow);
		r.col(minRow) = inf;
	}

	return r_new;
}*/