//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#pragma once

#include <iostream>
#include <stdlib.h>
#include <set>
#include "../../RaisimGymEnv.hpp"
#include "CPG.hpp"

#include <fstream>
#include <sstream>

//#define TEST_SIGNAL 1
#define INITIAL_SIG 1

namespace raisim {

    class ENVIRONMENT : public RaisimGymEnv {

    public:

        explicit ENVIRONMENT(const std::string &resourceDir, const Yaml::Node &cfg, bool visualizable) :
                RaisimGymEnv(resourceDir, cfg), visualizable_(visualizable), normDist_(0, 0.2) {
            /// create world
            world_ = std::make_unique<raisim::World>();

            /// add objects
            laikago_ = world_->addArticulatedSystem(resourceDir_ + "/laikago/laikago.urdf");
            laikago_->setName("laikago");
            laikago_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);

#ifndef TEST_SIGNAL
            world_->addGround();
#else
            outputOders();
#endif

            /// get robot data
            gcDim_ = laikago_->getGeneralizedCoordinateDim();
            gvDim_ = laikago_->getDOF();
            nJoints_ = gvDim_ - 6;

            /// initialize containers
            gc_.setZero(gcDim_);
            gc_init_.setZero(gcDim_);
            gv_.setZero(gvDim_);
            gv_init_.setZero(gvDim_);
            pTarget_.setZero(gcDim_);
            vTarget_.setZero(gvDim_);
            pTarget12_.setZero(nJoints_);

             // this is nominal configuration of laikago
            gc_init_ << 0, 0, 0.46, 1.0, 0.0, 0.0, 0.0, 0.5, -1, 0.5, -1, 0.5, -1, 0.5, -1.0;

            // cpg
            cpg_ = CPG<double>();
            // cpg_status
            cpg_status_.setZero(8);
            cpg_status_dot_.setZero(8);
            cpg_status_ = cpg_.get_status();

            /// set pd gains
            Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
            jointPgain.setZero();
            jointPgain.tail(nJoints_).setConstant(220.0); // 300
            jointDgain.setZero();
            jointDgain.tail(nJoints_).setConstant(10.0);
            laikago_->setPdGains(jointPgain, jointDgain);
            laikago_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

            /// MUST BE DONE FOR ALL ENVIRONMENTS
//             obDim_ = 34;
//            obDim_ = 26;
            obDim_ = 34;

            critic_obDim_ = 10; 
            actor_obDim_ = 8; 

            // actionDim_ = nJoints_
            actionDim_ = nJoints_;
            actionMean_.setZero(actionDim_);
            actionStd_.setZero(actionDim_);
            obDouble_.setZero(obDim_);

            /// action scaling
            actionMean_ = gc_init_.tail(nJoints_);
            actionStd_.setConstant(0.3);

            /// Reward coefficients
            rewards_.initializeFromConfigurationFile(cfg["reward"]);
            _torque_reward_coeff = cfg["reward"]["torque"]["coeff"].As<double>();
            _forward_reward_coeff = cfg["reward"]["forwardVel"]["coeff"].As<double>();
            _contact_reward_coeff = cfg["reward"]["contactReward"]["coeff"].As<double>();
            _yaw_reward_coeff = cfg["reward"]["yawReward"]["coeff"].As<double>();

            /// indices of links that should not make contact with ground
            footIndices_.insert(laikago_->getBodyIdx("FR_calf")); // 2
            footIndices_.insert(laikago_->getBodyIdx("FL_calf")); // 4
            footIndices_.insert(laikago_->getBodyIdx("RR_calf")); // 6
            footIndices_.insert(laikago_->getBodyIdx("RL_calf")); // 8

            /// visualize if it is the first environment
            if (visualizable_) {
                server_ = std::make_unique<raisim::RaisimServer>(world_.get());
                server_->launchServer();
                server_->focusOn(laikago_);
            }
            laikago_->setState(gc_init_, gv_init_);

            std::string cwd = getcwd(nullptr, 0);
            logfile = std::ofstream(cwd + "/data.txt");
        }

        void init() final {}

        void reset() final {
            cpg_.reset();
            laikago_->setState(gc_init_, gv_init_);

            episode_step = 0;
            updateObservation();

            _log_contact_reward = 0.0;
            _log_forward_reward = 0.0;
            _log_torque_reward = 0.0;
        }

        float step(const Eigen::Ref<EigenVec> &action, bool last_action=false) final {
            /// action scaling
            cpg_.step();
            cpg_status_ = cpg_.get_status();    
            pTarget12_ = action.cast<double>();
            pTarget12_ = pTarget12_.cwiseProduct(actionStd_);
            pTarget12_ += actionMean_;
            pTarget12_ += cpg_status_ * signal_scale_;

#ifdef TEST_SIGNAL
            gc_init_.tail(nJoints_) = pTarget12_;
            laikago_->setState(gc_init_, gv_init_);

            outputLog(cpg_status_, std::to_string(episode_step));
            auto raw_status_ = cpg_.get_raw_status();
            outputLog(raw_status_, std::to_string(episode_step));
#else
            pTarget_.tail(nJoints_) = pTarget12_;
            laikago_->setPdTarget(pTarget_, vTarget_);
#endif

            for (int i = 0; i < int(control_dt_ / simulation_dt_ + 1e-10); i++) {
                if (server_) server_->lockVisualizationServerMutex();
                world_->integrate();
                if (server_) server_->unlockVisualizationServerMutex();
            }

            updateObservation();

            double torque = laikago_->getGeneralizedForce().squaredNorm(); // squaredNorm
            double linVelReward = std::min(4.0, bodyLinearVel_[0]);
            double contactReward = calculate_contact_reward();
            double yawReward = 0.0;

            _log_torque_reward += torque;
            _log_contact_reward += contactReward;
            _log_forward_reward += linVelReward;

//            std::cout << episode_step << " " << _log_contact_reward << " " << (_log_contact_reward / (episode_step + 1)) * _contact_reward_coeff << std::endl;
            if (last_action) {
                yawReward = calculate_yaw_reward();
//                std::cout << "\n===========================================================" << std::endl;
//                std::cout << "total step " << episode_step << std::endl;
//                std::cout << "torque reward " << (_log_torque_reward / episode_step) * _torque_reward_coeff << std::endl;
//                // std::cout << "forward reward " << (_log_forward_reward / episode_step) * _forward_reward_coeff << std::endl;
//                std::cout << "contact reward " << (_log_contact_reward / (episode_step + 1)) * _contact_reward_coeff << std::endl;
//                std::cout << "yaw reward " << (yawReward / episode_step) * _yaw_reward_coeff << std::endl;
//                std::cout << "===========================================================\n" << std::endl;
            }

            RSISNAN_MSG(yawReward, gc_ << " is nan " << episode_step << " " << last_action)

            rewards_.record("torque", torque);
            rewards_.record("forwardVel", linVelReward);
            rewards_.record("contactReward", contactReward);
            rewards_.record("yawReward", yawReward);

            episode_step += 1;
            return rewards_.sum();
        }

        float calculate_contact_reward()
        {
            size_t numContact_ =  laikago_->getContacts().size();
            contactState_.setZero(4);
            signalState_.setZero(4);

            if (numContact_ > 0) {
                for (int k = 0; k < numContact_; k++) {
                    if (!laikago_->getContacts()[k].skip()) {
                        int idx = laikago_->getContacts()[k].getlocalBodyIndex();
                        if (idx == 2 && !contactState_[0])
                        {
                            contactState_[0] = 1.0;
                        }
                        else if (idx == 4 && !contactState_[1])
                        {
                            contactState_[1] = 1.0;
                        }
                        else if (idx == 6 && !contactState_[2])
                        {
                            contactState_[2] = 1.0;
                        }
                        else if (idx == 8 && !contactState_[3])
                        {
                            contactState_[3] = 1.0;
                        }
                    }
                }
            }

            float contact_reward_ = 0.0;
            // compute swing phase state
            for (int j = 0; j < 4; j++) {
                if (cpg_status_[2 * j + 1] == 0)
                    signalState_[j] = 1.0;
                if (cpg_status_[2 * j + 1] == 0 && contactState_[j] == 0)
                    contact_reward_ += 1.0;
            }

//            outputLog(signalState_, "signal");

            return contact_reward_;
        }

        float calculate_yaw_reward()
        {
            float yaw_reward_ = abs(gc_[1] / gc_[0]);
            if (isnan(yaw_reward_))
                yaw_reward_ = 0.0;
            return yaw_reward_;
        }

        void reward_logging(Eigen::Ref<EigenVec> rewards) final {}

        void position_logging(Eigen::Ref<EigenVec> positions) final {
            positions = gc_.head(3).cast<float>();
        }

        void contact_logging(Eigen::Ref<EigenBoolVec> contacts) final {
            size_t numContact_ =  laikago_->getContacts().size();
            for (int i = 0; i < 4; i++) {
                footContactState_[i] = false;
            }

            if (numContact_ > 0) {
                for (int k = 0; k < numContact_; k++) {
                    if (!laikago_->getContacts()[k].skip()) {
                        int idx = laikago_->getContacts()[k].getlocalBodyIndex();
                        if (idx == 2 && !footContactState_[0])
                        {
                            footContactState_[0] = true;
                        }
                        else if (idx == 4 && !footContactState_[1])
                        {
                            footContactState_[1] = true;
                        }
                        else if (idx == 6 && !footContactState_[2])
                        {
                            footContactState_[2] = true;
                        }
                        else if (idx == 8 && !footContactState_[3])
                        {
                            footContactState_[3] = true;
                        }
                    }
                }
            }

            contacts = footContactState_;
        }

        void set_target_velocity(Eigen::Ref<EigenVec> velocity) final {}

        void get_CPG_reward(Eigen::Ref<EigenVec> CPG_reward) final {}

        void updateObservation() {
            cpg_status_ = cpg_.get_status();   
            laikago_->getState(gc_, gv_);

            raisim::Vec<4> quat;
            raisim::Mat<3, 3> rot;
            quat[0] = gc_[3];
            quat[1] = gc_[4];
            quat[2] = gc_[5];
            quat[3] = gc_[6];
            raisim::quatToRotMat(quat, rot);

            bodyLinearVel_ = rot.e().transpose() * gv_.segment(0, 3);
            bodyAngularVel_ = rot.e().transpose() * gv_.segment(3, 3);

            obDouble_ << gc_[2], /// body height, 1
                    rot.e().row(2).transpose(), /// body orientation, 3
                    gc_.tail(8), /// joint angles, 8
                    bodyLinearVel_, bodyAngularVel_, /// body linear&angular velocity, 3, 3
                    gv_.tail(8), /// joint velocity
                    cpg_status_; /// CPG signal status
        }

        void observe(Eigen::Ref<EigenVec> ob) final {
            /// convert it to float
            ob = obDouble_.cast<float>();
        }

        void get_phase(Eigen::Ref<EigenVec> phase) final {
            /// convert it to float
            phase = cpg_status_.cast<float>();
        }

        float get_velocity() final {
            return bodyLinearVel_[0];
        }

        bool isTerminalState(float &terminalReward) final {
            terminalReward = float(terminalRewardCoeff_);

            /// if the contact body is not feet
            for(auto& contact: laikago_->getContacts())
              if(footIndices_.find(contact.getlocalBodyIndex()) == footIndices_.end())
                return true;

            terminalReward = 0.f;
            return false;
        }

        void curriculumUpdate() {};

        void outputOders() {
            std::vector<std::string> orders = laikago_->getMovableJointNames();
            std::cout << "Output joint orders: " << std::endl;
            int order_idx = 0;
            for (auto s: orders) {
                std::cout << (order_idx++) << ":" << s << std::endl;
            }
            std::cout << std::endl;
        }

        void change_gait(int gait=1) final {
            signal_scale_ = signal_scales[gait];
            gait_ = gait;
            cpg_.change_gait(gait);
        }

        void outputLog(Eigen::VectorXd &info, std::string name) {
            int log_len = info.size();
            for (int i = 0; i < log_len; i++) {
                logstr << info[i] << ((i == log_len-1) ? " " : ", ");
            }
            std::cout << name << ": " << logstr.str() << std::endl;
            logstr.str("");
        }

    private:
        int gait_ = 1;
        MaterialManager materials_;

        int gcDim_, gvDim_, nJoints_;
        bool visualizable_ = false;

        raisim::ArticulatedSystem *laikago_;
        Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, pTarget_, pTarget12_, vTarget_;
        double terminalRewardCoeff_ = -10.;
        Eigen::VectorXd actionMean_, actionStd_, obDouble_;
        Eigen::Vector3d bodyLinearVel_, bodyAngularVel_;
        std::set<size_t> footIndices_;

        /// these variables are not in use. They are placed to show you how to create a random number sampler.
        std::normal_distribution<double> normDist_;
        thread_local static std::mt19937 gen_;

        CPG<double> cpg_;
        Eigen::VectorXd cpg_status_;
        Eigen::VectorXd cpg_status_dot_;
        double signal_scale_=0.4;
        std::array<double, 4> signal_scales = {0.24, 0.2, 0.2, 0.24};

        std::stringstream logstr;
        std::ofstream logfile;
        std::array<std::string, 4> gait_names= {"WALK", "TROT", "PACE", "GALLOP"};
        Eigen::Matrix<bool, 4, 1> contact_status;
        Eigen::Matrix<bool, 4, 1> footContactState_;

        double _log_contact_reward = 0.0, _log_forward_reward = 0.0, _log_torque_reward = 0.0;
        int episode_step = 0;

        double _contact_reward_coeff, _yaw_reward_coeff, _forward_reward_coeff, _torque_reward_coeff;
        Eigen::VectorXd signalState_, contactState_;
    };

    thread_local std::mt19937 raisim::ENVIRONMENT::gen_;
}

