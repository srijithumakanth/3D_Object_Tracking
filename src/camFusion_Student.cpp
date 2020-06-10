
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2); // pixel coordinates
        pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}


void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            // draw individual point
            if(it2==it1->lidarPoints.begin())
            {
              cv::circle(topviewImg, cv::Point(x, y), 6, cv::Scalar(0, 0, 255), -1);
            }
            else
            {
                cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);

            }
            
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(1); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    // calculate mean point match distance
    double meanDistance = 0.0;
    double size = 0.0;

    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end();  ++it1)
    {
        cv::KeyPoint currPoint = kptsCurr[it1->trainIdx];
        cv::KeyPoint prevPoint = kptsPrev[it1->queryIdx];

        if (boundingBox.roi.contains(currPoint.pt))
        {
            meanDistance += cv::norm(currPoint.pt - prevPoint.pt);
            size += 1;
        }
    }
    meanDistance = meanDistance / size;
    cout << " meanDistance: " << meanDistance << std::endl;
    

    // filter out points based on the mean distance
    for (auto it2 = kptMatches.begin(); it2 != kptMatches.end(); ++it2)
    {
        cv::KeyPoint currPoint = kptsCurr[it2->trainIdx];
        cv::KeyPoint prevPoint = kptsPrev[it2->queryIdx];

        if (boundingBox.roi.contains(currPoint.pt))
        {
            double currDistance = cv::norm(currPoint.pt - prevPoint.pt);
            double scaledMeanDistance = meanDistance * 1.3;
            // cout << " currDistance: " << currDistance << std::endl;
            // cout << " scaledMeanDistance: " << scaledMeanDistance << std::endl;

            if (currDistance < scaledMeanDistance)
            {
                boundingBox.keypoints.push_back(currPoint);
                boundingBox.kptMatches.push_back(*it2);
                // cout << " BBox kptsMatches.size(): " << boundingBox.kptMatches.size() << std::endl;
            }
        }
    }
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // calculate sitance ratio between all matches
    vector<double> distRatios;
    double minDist = 100.0; // minimum required distance
    double dT = 1 / frameRate;

    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end(); ++it1)
    {
        cv::KeyPoint currPointOuter = kptsCurr[it1->trainIdx];
        // cv::KeyPoint currPointOuter = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint prevPointOuter = kptsPrev[it1->queryIdx];
        // cv::KeyPoint prevPointOuter = kptsPrev.at(it1->queryIdx);

        for (auto it2 = kptMatches.begin(); it2 != kptMatches.end(); ++it2)
        {
            // calculate the current distance
            cv::KeyPoint currPointInner = kptsCurr[it2->trainIdx];
            // cv::KeyPoint currPointInner = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint prevPointInner = kptsPrev[it2->queryIdx];
            // cv::KeyPoint prevPointInner = kptsPrev.at(it2->queryIdx);

            double distCurr = cv::norm(currPointOuter.pt - currPointInner.pt);
            double distPrev = cv::norm(prevPointOuter.pt - prevPointInner.pt);

            if (distPrev >  std::numeric_limits<double>::epsilon() && distCurr >= minDist)
            {
                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        } // eof inner loop
    } // eof outer loop

    if (distRatios.size() == 0)
    {
        TTC = NAN;
        return;
    }

    std::sort(distRatios.begin(), distRatios.end());
    long medianIndex = floor(distRatios.size() / 2.0);

    double medianDistRatio = distRatios.size() % 2 == 0 ? (distRatios[medianIndex - 1] + distRatios[medianIndex]) / 2 : distRatios[medianIndex];

    TTC = -dT / (1 - medianDistRatio); 
}

// performing eucledian clustering to remove outliers in LiDAR data
pcl::PointCloud<pcl::PointXYZ>::Ptr removeLidarOutliers (std::vector<LidarPoint>& lidarPoints)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr result(new pcl::PointCloud<pcl::PointXYZ>);
    
    for (const auto&pt : lidarPoints)
    {
        cloud->push_back(pcl::PointXYZ((float)pt.x, (float)pt.y, 0.0f));
    }

    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(cloud);

    std::vector<pcl::PointIndices> clusterIndices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance (0.05); 
    ec.setMinClusterSize (3); // 3
    // ec.setMaxClusterSize (15);
    ec.setSearchMethod (tree);
    ec.setInputCloud (cloud);
    ec.extract (clusterIndices);

    if (clusterIndices.empty())
    {
        return result;
    }

    for (auto& getIndices : clusterIndices)
    {
        for (int i : getIndices.indices)
        {
            result->points.push_back(cloud->points.at(i));
        }
    }

    return result;
}

void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    double dT = 1 / frameRate;
    double minPrevX = 1e9;
    double minCurrX = 1e9;

    pcl::PointCloud<pcl::PointXYZ>::Ptr prevLidarCloud = removeLidarOutliers(lidarPointsPrev);
    pcl::PointCloud<pcl::PointXYZ>::Ptr currLidarCloud = removeLidarOutliers(lidarPointsCurr);

    for (const auto& pt : prevLidarCloud->points)
    {
        minPrevX = minPrevX > pt.x ? pt.x : minPrevX;
    }

    for (const auto& pt : currLidarCloud->points)
    {
        minCurrX = minCurrX > pt.x ? pt.x : minCurrX;
    }

    TTC = (minCurrX * dT) / (minPrevX - minCurrX);
}


void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    std::map <std::pair<int,int>, int> bbMatchCounter; // [(bbInPrevID, bbInCurreID), numOfOccurences]

    for (auto& match : matches) // loop through all the matched descriptors
    {
        // cv::KeyPoint prevPoint = prevFrame.keypoints[match.queryIdx];
        cv::Point prevPoint = prevFrame.keypoints.at(match.queryIdx).pt;
        // cv::KeyPoint currPoint = currFrame.keypoints[match.trainIdx];
        cv::Point currPoint = currFrame.keypoints.at(match.trainIdx).pt;

        for(auto& bboxInPrev : prevFrame.boundingBoxes)
        {
            // if (!bboxInPrev.roi.contains(prevPoint.pt))
            if (!bboxInPrev.roi.contains(prevPoint))
            {
                continue;
            }

            for (auto& bboxInCurr : currFrame.boundingBoxes)
            {
                // if(!bboxInCurr.roi.contains(currPoint.pt))
                if(!bboxInCurr.roi.contains(currPoint))
                {
                    continue;
                }

                bbMatchCounter[std::make_pair(bboxInPrev.boxID, bboxInCurr.boxID)]++;
            }
        }

        std::vector<std::tuple<int, int, int>> bboxMatches;

        // lambda expresseion to add prevBoxID, currBoxID, numOfOccurences into bboxMatches vector of tuples
        std::for_each(bbMatchCounter.begin(), bbMatchCounter.end(), [&bboxMatches](std::pair<std::pair<int,int>, int> pair)
        {
            bboxMatches.emplace_back(pair.first.first, pair.first.second, pair.second);
        });

        // lambda expression to sort the vector of tuples in descending order of numOfOccurences
        std::sort(bboxMatches.begin(), bboxMatches.end(), [](const std::tuple<int, int, int>&a, const std::tuple<int,int,int>&b )
        {
            return (std::get<2>(a) > std::get<2>(b));
        });

        std::set<int> matchedPrevBoxes; // to avoid already matched previous bounding boxes
        
        bbBestMatches.clear();
        for (auto& t : bboxMatches)
        {
            if (matchedPrevBoxes.count(std::get<0>(t))) // If already matched bbox exist, then continue(to avoid repetation)
            {
                continue;
            }

            matchedPrevBoxes.insert(std::get<0>(t));
            bbBestMatches.insert(std::make_pair(std::get<0>(t), std::get<1>(t)));
            // std::cout << "Matching Bounding Boxes Complete"  << std::endl;

        }
    }
}
