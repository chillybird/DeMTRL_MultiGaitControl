#ifndef CPG_HPP
#define CPG_HPP

#include <set>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Dense>

#include <cstdlib>
#include <ctime>

#include "reference_walk.hpp"
#include "reference_trot.hpp"
#include "reference_pace.hpp"
#include "reference_bound.hpp"

// #define RSI_SET 1

namespace raisim
{

    template<typename T>
    class CPG
    {
    public:
        CPG()
        {
            // 0: WALK; 1: TROT; 2: PACE; 3: GALLOP; -1: STAND
            OFFSET[0] = {0.0, M_PI, M_PI * 0.5, M_PI * 1.5};
            OFFSET[1] = {0.0, M_PI, M_PI, 0.0};
            OFFSET[2] = {0.0, M_PI, 0.0, M_PI};
            OFFSET[3] = {0.0, 0.0, M_PI, M_PI};

            srand(time(NULL));

            reset_gait_four_leg(gait_);
        }

        inline void update_r_mat_four_leg()
        {
            r_mat_.setZero();

            for (int i = 0; i < 4; ++i)
            {
                for (int j = 0; j < 4; ++j)
                {
                    T theta = offset_[i] - offset_[j];
                    r_mat_(2 * j, 2 * i) = std::cos(theta);
                    r_mat_(2 * j + 1, 2 * i + 1) = std::cos(theta);
                    r_mat_(2 * j, 2 * i + 1) = -std::sin(theta);
                    r_mat_(2 * j + 1, 2 * i) = std::sin(theta);
                }
            }
        }

        inline void reset_gait_four_leg(int gait)
        {
            gait_ = gait;
            beta_ = BETA[gait_];
            delta_ = DELTA[gait_];
            t_ = TIME[gait_];
            offset_ = OFFSET[gait_];
            phase_ = offset_;

            Wst = ((1 - beta_) / beta_) * Wsw;

#ifdef RSI_SET

            if (gait == 0)
            {
                rand_idx = rand() % walk_data.size();
                for (int i = 0; i < walk_data[0].size(); i++)
                    q_(i, 0) = walk_data[rand_idx][i];
            }
            else if (gait == 1)
            {
                rand_idx = rand() % trot_data.size();
                for (int i = 0; i < trot_data[0].size(); i++)
                    q_(i, 0) = trot_data[rand_idx][i];
            }
            else if (gait == 2)
            {
                rand_idx = rand() % pace_data.size();
                for (int i = 0; i < pace_data[0].size(); i++)
                    q_(i, 0) = pace_data[rand_idx][i];
            }
            else
            {
                rand_idx = rand() % bound_data.size();
                for (int i = 0; i < bound_data[0].size(); i++)
                    q_(i, 0) = bound_data[rand_idx][i];
            }
#else
            for (int i = 0; i < 4; ++i)
            {
                q_[2 * i] = std::sin(phase_[i]);
                q_[2 * i + 1] = std::cos(phase_[i]);
            }
#endif

            update_r_mat_four_leg();
        }

        inline void change_gait(int gait)
        {
            reset_gait_four_leg(gait);
        }

        inline void step()
        {

            f_mat_.setZero();
            for (int i = 0; i < 4; ++i)
            {
                r_square_ = q_[2 * i] * q_[2 * i] + q_[2 * i + 1] * q_[2 * i + 1];
                f_mat_(2 * i, 2 * i) = alpha_ * (mu_ - r_square_);
                f_mat_(2 * i + 1, 2 * i + 1) = gamma_ * (mu_ - r_square_);
                T omega = M_PI / t_ * (1.0 / beta_ / (1.0 + std::exp(-b_ * q_(2 * i + 1))) + 1.0 / (1.0 - beta_) / (1.0 + std::exp(b_ * q_(2 * i + 1))));
//                T omega = (Wst / (std::exp(-b_ * q_(2 * i + 1)) + 1)) + (Wsw / (std::exp(b_ * q_(2 * i + 1)) + 1));
                f_mat_(2 * i, 2 * i + 1) = -omega;
                f_mat_(2 * i + 1, 2 * i) = omega;
            }

            q_dot_ = f_mat_ * q_ + r_mat_ * q_ * delta_;
            q_ = q_ + q_dot_ * dt_;
        }

        inline void reset()
        {
            reset_gait_four_leg(gait_);
        }

        Eigen::Matrix<T, Eigen::Dynamic, 1> get_status()
        {
            q_cp = q_;
            for (int i = 0; i < 8; i++) {
                if (i % 2 != 0) {
                    if (q_cp[i] > 0)
                        q_cp[i] = 0;
                    else
                        q_cp[i] = -q_cp[i];
                }
            }

            return -q_cp;
        }

        Eigen::Matrix<T, Eigen::Dynamic, 1> get_raw_status()
        {
            return q_;
        }

        Eigen::Matrix<T, Eigen::Dynamic, 1> get_velocity()
        {
            return q_dot_;
        }

        int get_gait_index()
        {
            return gait_;
        }

        // private:
        T mu_ = 1.0;
        T alpha_ = 50.0;
        T gamma_ = 50.0;
        T b_ = 50.0;
        T dt_ = 0.01;

        T r_square_;
        int rand_idx=0;

        std::array<T, 4> BETA = {0.75, 0.5, 0.5, 0.4};
        std::array<T, 4> DELTA = {1.0, 1.0, 1.0, 1.0};
        std::array<T, 4> TIME = {0.6, 0.5, 0.5, 0.3};
        std::array<std::array<T, 4>, 4> OFFSET;

        double Wst = 0.0, Wsw = 4 * M_PI;

        int gait_ = 0;

        T beta_, delta_, t_;
        std::array<T, 4> offset_;
        std::array<T, 4> phase_; // 0 to 2 * pi

        Eigen::Matrix<T, 8, 1> q_, q_cp;
        Eigen::Matrix<T, 8, 1> q_dot_;
        Eigen::Matrix<T, 8, 8> r_mat_;
        Eigen::Matrix<T, 8, 8> f_mat_;
        Eigen::Matrix<T, 4, 1> phase_raw_; // -pi to +pi
        Eigen::Matrix<T, 4, 1> omega_;

    };

} // namespace raisim

#endif