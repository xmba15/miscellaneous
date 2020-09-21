/**
 * @file    draw_coord_app.cpp
 *
 * @author  btran
 *
 */

#include <iostream>
#include <thread>

#include <pangolin/pangolin.h>

namespace
{
using Pose = Eigen::Isometry3d;
using Poses = std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>>;

Eigen::Quaterniond toQuaternion(const double roll, const double pitch, const double yaw)
{
    Eigen::AngleAxisd rollAngle(roll, Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd pitchAngle(pitch, Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd yawAngle(yaw, Eigen::Vector3d::UnitZ());

    Eigen::Quaterniond q = yawAngle * pitchAngle * rollAngle;
    return q;
}

void drawPoses(const Poses& poses);
}  // namespace

int main(int argc, char* argv[])
{
    if (argc != 4) {
        std::cerr << "Usage: [app] [show/default] [use/quat] [optional/pose/file]" << std::endl;
        return EXIT_FAILURE;
    }

    bool showDefault = std::atoi(argv[1]);
    bool useQuat = std::atoi(argv[2]);
    std::string posesPath = argv[3];

    Poses poses;

    if (showDefault) {
        for (float i = 0; i < 20; i += 0.5) {
            Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
            pose.translation() << i, 0, 0;
            poses.emplace_back(pose);
        }
    } else {
        std::ifstream fin(posesPath);
        while (!fin.eof()) {
            if (useQuat) {
                double tx, ty, tz, qx, qy, qz, qw;
                fin >> tx >> ty >> tz >> qx >> qy >> qz >> qw;
                Eigen::Isometry3d pose(Eigen::Quaternion(qw, qx, qy, qz));
                pose.pretranslate(Eigen::Vector3d(tx, ty, tz));
                poses.emplace_back(pose);
            } else {
                double tx, ty, tz, r, p, y;
                fin >> tx >> ty >> tz >> r >> p >> y;
                Eigen::Isometry3d pose(toQuaternion(r, p, y));
                pose.pretranslate(Eigen::Vector3d(tx, ty, tz));
                poses.emplace_back(pose);
            }
        }
    }
    ::drawPoses(poses);

    return EXIT_SUCCESS;
}

namespace
{
void drawPoses(const Poses& poses)
{
    if (poses.empty()) {
        return;
    }

    pangolin::CreateWindowAndBind("Trajectory Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState sCam(pangolin::ProjectionMatrix(1924, 768, 500, 500, 512, 389, 0.1, 1000),
                                     pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0, -1, 0));

    pangolin::View& dCam =
        pangolin::CreateDisplay().SetBounds(0, 1, 0, 1, -1024. / 768).SetHandler(new pangolin::Handler3D(sCam));

    while (pangolin::ShouldQuit() == false) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        dCam.Activate(sCam);
        glClearColor(1., 1., 1., 1.);
        glLineWidth(2);

        for (const auto& pose : poses) {
            Eigen::Vector3d Ow = pose.translation();
            Eigen::Vector3d Xw = pose * (0.1 * Eigen::Vector3d(1, 0, 0));
            Eigen::Vector3d Yw = pose * (0.1 * Eigen::Vector3d(0, 1, 0));
            Eigen::Vector3d Zw = pose * (0.1 * Eigen::Vector3d(0, 0, 1));
            glBegin(GL_LINES);
            glColor3f(1., 0., 0.);
            glVertex3d(Ow[0], Ow[1], Ow[2]);
            glVertex3d(Xw[0], Xw[1], Xw[2]);
            glColor3f(0., 1., 0.);
            glVertex3d(Ow[0], Ow[1], Ow[2]);
            glVertex3d(Yw[0], Yw[1], Yw[2]);
            glColor3f(0., 0., 1.);
            glVertex3d(Ow[0], Ow[1], Ow[2]);
            glVertex3d(Zw[0], Zw[1], Zw[2]);
            glEnd();
        }

        for (std::size_t i = 0; i < poses.size() - 1; ++i) {
            glColor3f(0.0, 0.0, 0.0);
            glBegin(GL_LINES);
            const Pose& p1 = poses[i];
            const Pose& p2 = poses[i + 1];
            glVertex3d(p1.translation()[0], p1.translation()[1], p1.translation()[2]);
            glVertex3d(p2.translation()[0], p2.translation()[1], p2.translation()[2]);
            glEnd();
        }
        pangolin::FinishFrame();
        std::this_thread::sleep_for(std::chrono::microseconds(5000));
    }
}
}  // namespace
