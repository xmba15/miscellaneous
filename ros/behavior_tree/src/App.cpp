/**
 * @file    App.cpp
 *
 * @author  btran
 *
 */

#include <behaviortree_cpp_v3/bt_factory.h>

namespace
{
BT::NodeStatus CheckBattery()
{
    std::cout << "[ Battery: OK ]" << std::endl;
    return BT::NodeStatus::SUCCESS;
}

class ApproachObject : public BT::SyncActionNode
{
 public:
    ApproachObject(const std::string& name) : BT::SyncActionNode(name, {})
    {
    }

    BT::NodeStatus tick() override
    {
        std::cout << "ApproachObject: " << this->name() << std::endl;
        return BT::NodeStatus::SUCCESS;
    }
};

class GripperInterface
{
 public:
    GripperInterface() : _open(true)
    {
    }

    BT::NodeStatus open()
    {
        _open = true;
        std::cout << "GripperInterface::open" << std::endl;
        return BT::NodeStatus::SUCCESS;
    }

    BT::NodeStatus close()
    {
        std::cout << "GripperInterface::close" << std::endl;
        _open = false;
        return BT::NodeStatus::SUCCESS;
    }

 private:
    bool _open;  // shared information
};
}  // namespace

int main(int argc, char* argv[])
{
    if (argc != 2) {
        std::cerr << "Usage: [app] [path/to/xml/file]" << std::endl;
        return EXIT_FAILURE;
    }

    const std::string xmlPath = argv[1];

    BT::BehaviorTreeFactory factory;

    factory.registerNodeType<ApproachObject>("ApproachObject");
    factory.registerSimpleCondition("CheckBattery", std::bind(CheckBattery));
    GripperInterface gripper;
    factory.registerSimpleAction("OpenGripper", std::bind(&GripperInterface::open, &gripper));
    factory.registerSimpleAction("CloseGripper", std::bind(&GripperInterface::close, &gripper));
    auto tree = factory.createTreeFromFile(xmlPath);

    tree.tickRoot();

    return EXIT_SUCCESS;
}
