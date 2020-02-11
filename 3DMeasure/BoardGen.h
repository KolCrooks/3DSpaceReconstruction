#pragma once
#include <opencv2/aruco/charuco.hpp>


class Board {
public:
	static struct BoardProps {
	public:
		const int chessX = 4;
		const int chessY = 4;
		const float sqrSize = 0.05f;
		const float mrkSize = 0.025f;
	};

	const enum Surface{
		WALL,
		FLOOR
	};

	cv::Ptr<cv::aruco::CharucoBoard> charBoard;
	cv::Mat boardImage;
	static cv::Ptr<cv::aruco::Dictionary> dictionary;
	Board(Surface surface = FLOOR);
	bool detect(cv::Mat frame, cv::InputArray cameraMatrix, cv::InputArray distCoeffs, cv::Vec3d rvec, cv::Vec3d tvec, cv::Mat drawTo = cv::Mat());
private:
	static size_t offset;
};