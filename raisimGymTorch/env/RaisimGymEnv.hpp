//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#ifndef SRC_RAISIMGYMENV_HPP
#define SRC_RAISIMGYMENV_HPP

#include <vector>
#include <memory>
#include <unordered_map>
#include "Common.hpp"
#include "raisim/World.hpp"
#include "raisim/RaisimServer.hpp"
#include "Yaml.hpp"
#include "Reward.hpp"

namespace raisim {


    class RaisimGymEnv {

    public:
        explicit RaisimGymEnv(std::string resourceDir, const Yaml::Node &cfg) :
                resourceDir_(std::move(resourceDir)), cfg_(cfg) {}

        virtual ~RaisimGymEnv() { if (server_) server_->killServer(); };

        /////// implement these methods /////////
        virtual void init() = 0;

        virtual void reset() = 0;

        // virtual void episodeReset() = 0;
        virtual void observe(Eigen::Ref<EigenVec> ob) = 0;

        virtual float step(const Eigen::Ref<EigenVec> &action, bool last_action=false) = 0;

        virtual bool isTerminalState(float &terminalReward) = 0;
        ////////////////////////////////////////

        /////// optional methods ///////
        virtual void curriculumUpdate() {};

        virtual void close() {};

        virtual void setSeed(int seed) {};

        virtual void reward_logging(Eigen::Ref<EigenVec> rewards) = 0;

        virtual void position_logging(Eigen::Ref<EigenVec> positions) = 0;

        virtual void contact_logging(Eigen::Ref<EigenBoolVec> contacts) = 0;

        virtual void set_target_velocity(Eigen::Ref<EigenVec> velocity) = 0;

        virtual void get_CPG_reward(Eigen::Ref<EigenVec> CPG_reward) = 0;
        // virtual void calculate_cost() = 0;
        // virtual void comprehend_contacts() = 0;

        // define my methods
        virtual void change_gait(int gait) = 0;
        virtual void get_phase(Eigen::Ref<EigenVec> ob) = 0; // get gait phase signal
        virtual float get_velocity() = 0; // get velocity from envs
//  virtual void ma_step(const Eigen::Ref<EigenVec>& action) = 0;
//  virtual void ma_reset() = 0;

        ////////////////////////////////

        void setSimulationTimeStep(double dt) {
            simulation_dt_ = dt;
            world_->setTimeStep(dt);
        }

        void setControlTimeStep(double dt) { control_dt_ = dt; }

        int getObDim() { return obDim_; }

        int getCriticObDim() { return critic_obDim_; }

        int getActorObDim() { return actor_obDim_; }

        int getActionDim() { return actionDim_; }

        double getControlTimeStep() { return control_dt_; }

        double getSimulationTimeStep() { return simulation_dt_; }

        raisim::World *getWorld() { return world_.get(); }

        void turnOffVisualization() { server_->hibernate(); }

        void turnOnVisualization() { server_->wakeup(); }

        void startRecordingVideo(const std::string &videoName) { server_->startRecordingVideo(videoName); }

        void stopRecordingVideo() { server_->stopRecordingVideo(); }

        raisim::Reward &getRewards() { return rewards_; }

    protected:
        std::unique_ptr<raisim::World> world_;
        double simulation_dt_ = 0.001;
        double control_dt_ = 0.01;
        std::string resourceDir_;
        Yaml::Node cfg_;
        int obDim_ = 0, actionDim_ = 0, critic_obDim_ = 0, actor_obDim_;
        std::unique_ptr<raisim::RaisimServer> server_;
        raisim::Reward rewards_;
    };
}

#endif //SRC_RAISIMGYMENV_HPP
