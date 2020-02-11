#include "BoardGen.h"

Board::Board(Surface surface) {
	BoardProps boardProps;
	charBoard = cv::aruco::CharucoBoard::create(boardProps.chessX, boardProps.chessY, boardProps.sqrSize, boardProps.mrkSize, dictionary);
	for (int& id : charBoard->ids) {
		id += (int)offset;
	}
	charBoard->draw(cv::Size(600, 500), boardImage, 10, 1);
	offset += charBoard->ids.size();
}

bool Board::detect(cv::Mat frame, cv::InputArray cameraMatrix, cv::InputArray distCoeffs, cv::Vec3d rvec, cv::Vec3d tvec, cv::Mat drawTo) {
    bool doDraw = !drawTo.empty();
    std::vector<int> markerIds;
    std::vector<std::vector<cv::Point2f>> markerCorners;
    cv::aruco::detectMarkers(frame, dictionary, markerCorners, markerIds);

    std::vector<cv::Point2f> charucoCorners;
    std::vector<int> charucoIds;

    if (markerIds.size() > 0) {
        cv::aruco::interpolateCornersCharuco(markerCorners, markerIds, frame, charBoard, charucoCorners, charucoIds, cameraMatrix, distCoeffs);
        if (charucoIds.size() > 0) {
            if(doDraw)
                cv::aruco::drawDetectedCornersCharuco(drawTo, charucoCorners, charucoIds);
            bool valid = cv::aruco::estimatePoseCharucoBoard(charucoCorners, charucoIds, charBoard, cameraMatrix, distCoeffs, rvec, tvec);
            if (valid) {
                if (doDraw)
                    cv::aruco::drawAxis(drawTo, cameraMatrix, distCoeffs, rvec, tvec, 0.1f);
                return true;
            }
        }
    }
    return false;
}

size_t Board::offset = 0;
cv::Ptr<cv::aruco::Dictionary> Board::dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_250);
