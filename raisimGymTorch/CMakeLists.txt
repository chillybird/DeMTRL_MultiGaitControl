cmake_minimum_required(VERSION 3.21)
project(raisimGymTorch)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)
include_directories(env/envs)
include_directories(env/envs/rsg_anymal)

add_executable(raisimGymTorch
        env/envs/rsg_anymal/Environment.hpp
        env/Common.hpp
        env/debug_app.cpp
        env/raisim_gym.cpp
        env/RaisimGymEnv.hpp
        env/Reward.hpp
        env/VectorizedEnvironment.hpp
        env/Yaml.cpp
        env/Yaml.hpp
        eigen_use.cpp)


add_executable(eigen_use eigen_use.cpp)