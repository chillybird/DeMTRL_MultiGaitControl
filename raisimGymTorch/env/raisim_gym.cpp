//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "Environment.hpp"
#include "VectorizedEnvironment.hpp"

namespace py = pybind11;
using namespace raisim;
int THREAD_COUNT = 1;

#ifndef ENVIRONMENT_NAME
  #define ENVIRONMENT_NAME RaisimGymEnv
#endif

PYBIND11_MODULE(RAISIMGYM_TORCH_ENV_NAME, m) {
  py::class_<VectorizedEnvironment<ENVIRONMENT>>(m, RSG_MAKE_STR(ENVIRONMENT_NAME))
    .def(py::init<std::string, std::string>(), py::arg("resourceDir"), py::arg("cfg"))
    .def("init", &VectorizedEnvironment<ENVIRONMENT>::init)
    .def("reset", &VectorizedEnvironment<ENVIRONMENT>::reset)
    // .def("episodeReset", &VectorizedEnvironment<ENVIRONMENT>::episodeReset)
    .def("observe", &VectorizedEnvironment<ENVIRONMENT>::observe)
    .def("step", &VectorizedEnvironment<ENVIRONMENT>::step)
    .def("setSeed", &VectorizedEnvironment<ENVIRONMENT>::setSeed)
    .def("rewardInfo", &VectorizedEnvironment<ENVIRONMENT>::getRewardInfo)
    .def("close", &VectorizedEnvironment<ENVIRONMENT>::close)
    .def("isTerminalState", &VectorizedEnvironment<ENVIRONMENT>::isTerminalState)
    .def("setSimulationTimeStep", &VectorizedEnvironment<ENVIRONMENT>::setSimulationTimeStep)
    .def("setControlTimeStep", &VectorizedEnvironment<ENVIRONMENT>::setControlTimeStep)
    .def("getObDim", &VectorizedEnvironment<ENVIRONMENT>::getObDim)
    .def("getActionDim", &VectorizedEnvironment<ENVIRONMENT>::getActionDim)
    .def("getNumOfEnvs", &VectorizedEnvironment<ENVIRONMENT>::getNumOfEnvs)
    .def("turnOnVisualization", &VectorizedEnvironment<ENVIRONMENT>::turnOnVisualization)
    .def("turnOffVisualization", &VectorizedEnvironment<ENVIRONMENT>::turnOffVisualization)
    .def("stopRecordingVideo", &VectorizedEnvironment<ENVIRONMENT>::stopRecordingVideo)
    .def("startRecordingVideo", &VectorizedEnvironment<ENVIRONMENT>::startRecordingVideo)
    .def("curriculumUpdate", &VectorizedEnvironment<ENVIRONMENT>::curriculumUpdate)
    .def("getObStatistics", &VectorizedEnvironment<ENVIRONMENT>::getObStatistics)
    .def("setObStatistics", &VectorizedEnvironment<ENVIRONMENT>::setObStatistics)
    .def("reward_logging", &VectorizedEnvironment<ENVIRONMENT>::reward_logging)
    .def("position_logging", &VectorizedEnvironment<ENVIRONMENT>::position_logging)
    .def("contact_logging", &VectorizedEnvironment<ENVIRONMENT>::contact_logging)
    .def("set_target_velocity", &VectorizedEnvironment<ENVIRONMENT>::set_target_velocity)
    .def("get_CPG_reward", &VectorizedEnvironment<ENVIRONMENT>::get_CPG_reward)
    // .def("increase_cost_scale", &VectorizedEnvironment<ENVIRONMENT>::increase_cost_scale)
    // .def("calculate_cost", &VectorizedEnvironment<ENVIRONMENT>::calculate_cost)
    // .def("comprehend_contacts", &VectorizedEnvironment<ENVIRONMENT>::comprehend_contacts)

    // define my methods
    .def("change_gait", &VectorizedEnvironment<ENVIRONMENT>::change_gait)
//    .def("ma_step", &VectorizedEnvironment<ENVIRONMENT>::ma_step)
    .def("ma_reset", &VectorizedEnvironment<ENVIRONMENT>::ma_reset)
    .def("getCriticObDim", &VectorizedEnvironment<ENVIRONMENT>::getCriticObDim)
    .def("getActorObDim", &VectorizedEnvironment<ENVIRONMENT>::getActorObDim)
    .def("get_phase", &VectorizedEnvironment<ENVIRONMENT>::get_phase)
    .def("get_velocity", &VectorizedEnvironment<ENVIRONMENT>::get_velocity)

    .def(py::pickle(
        [](const VectorizedEnvironment<ENVIRONMENT> &p) { // __getstate__ --> Pickling to Python
            /* Return a tuple that fully encodes the state of the object */
            return py::make_tuple(p.getResourceDir(), p.getCfgString());
        },
        [](py::tuple t) { // __setstate__ - Pickling from Python
            if (t.size() != 2) {
              throw std::runtime_error("Invalid state!");
            }

            /* Create a new C++ instance */
            VectorizedEnvironment<ENVIRONMENT> p(t[0].cast<std::string>(), t[1].cast<std::string>());

            return p;
        }
    ));

    py::class_<NormalSampler>(m, "NormalSampler")
    .def(py::init<int>(), py::arg("dim"))
    .def("seed", &NormalSampler::seed)
    .def("sample", &NormalSampler::sample);
}
