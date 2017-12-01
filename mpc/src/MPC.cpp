#include "MPC.h"
#include <cppad/cppad.hpp>
#include <cppad/ipopt/solve.hpp>
#include "Eigen-3.3/Eigen/Core"

using CppAD::AD;

#define COST_CROSSTRACK_ERROR 100 // How bad is it to be away from the centre of the road? Pretty bad.
#define COST_ANGLE_ERROR      10  // How bad is it to be angled differently than the ideal path? Not as bad.
#define COST_SPEED_ERROR      5   // How bad is it to be going slower than we want? Okay.
#define COST_MOVEMENT_ERROR   10  // How bad is it to speed up or turn left or right quickly? Bad.
#define COST_JERK_ERROR       20  // How bad is it to speed up, slow down, turn left and right jittery? Real bad.
#define COST_SPEED_CORNERING  50  // How bad is it to have velocity while taking a corner? Just don't do it!


// Set the timestep length and duration
size_t N = 10;
double dt = 0.5;

// This value assumes the model presented in the classroom is used.
//
// It was obtained by measuring the radius formed by running the vehicle in the
// simulator around in a circle with a constant steering angle and velocity on a
// flat terrain.
//
// Lf was tuned until the the radius formed by the simulating the model
// presented in the classroom matched the previous radius.
//
// This is the length from front to CoG that has a similar radius.
const double Lf = 2.67;

// How fast should we aim to go?
double ref_v = 20;

// The solver takes all the state variables and actuator variables in a singular vector.
// We should to establish when one variable starts and another ends to make our life easier.
size_t x_start = 0;
size_t y_start = x_start + N;
size_t psi_start = y_start + N;
size_t v_start = psi_start + N;
size_t cte_start = v_start + N;
size_t epsi_start = cte_start + N;
size_t delta_start = epsi_start + N;  // These two are special.
size_t a_start = delta_start + N - 1; // They are the steering and acceleration output.

class FG_eval {
 public:
  // Fitted polynomial coefficients
  Eigen::VectorXd coeffs;
  FG_eval(Eigen::VectorXd coeffs) { this->coeffs = coeffs; }

  typedef CPPAD_TESTVECTOR(AD<double>) ADvector;

  // `fg` is a vector containing the cost and constraints.
  // `vars` is a vector containing the variable values (state & actuators).
  void operator()(ADvector& fg, const ADvector& vars) {

    // Our cost function
    fg[0] = 0;

    // The part of the cost based on the reference state.
    // + Crosstrack error ^2
    // + Error psi angle ^2
    // + Velocity below target speed ^2

    for (int t = 0; t < N; t++) {
      fg[0] += COST_CROSSTRACK_ERROR * CppAD::pow(vars[cte_start + t], 2);
      fg[0] += COST_ANGLE_ERROR * CppAD::pow(vars[epsi_start + t], 2);
      fg[0] += COST_SPEED_ERROR * CppAD::pow(vars[v_start + t] - ref_v, 2);
    }

    // Minimize the use of actuators.
    // Add cost for:
    // + Changing delta steering angle
    // + Increasing or decreasing acceleration
    // + Having speed while cornering
    for (int t = 0; t < N - 1; t++) {
      fg[0] += COST_MOVEMENT_ERROR * CppAD::pow(vars[delta_start + t], 2);
      fg[0] += COST_MOVEMENT_ERROR * CppAD::pow(vars[a_start + t], 2);
      fg[0] += COST_SPEED_CORNERING * CppAD::pow(vars[delta_start + t] * vars[v_start + t], 2);
    }

    // Minimize the value gap between sequential actuations.
    // The next action should look like the current action!
    for (int t = 0; t < N - 2; t++) {
      fg[0] += COST_JERK_ERROR * CppAD::pow(vars[delta_start + t + 1] - vars[delta_start + t], 2);
      fg[0] += COST_JERK_ERROR * CppAD::pow(vars[a_start + t + 1] - vars[a_start + t], 2);
    }

    // Initial constraints
    // We add 1 to each of the starting indices due to cost being located at index 0 of `fg`.
    // This bumps up the position of all the other values.
    fg[1 + x_start] = vars[x_start];
    fg[1 + y_start] = vars[y_start];
    fg[1 + psi_start] = vars[psi_start];
    fg[1 + v_start] = vars[v_start];
    fg[1 + cte_start] = vars[cte_start];
    fg[1 + epsi_start] = vars[epsi_start];

    // The rest of the constraints
    for (int t = 1; t < N; t++) {
      // The state at time t+1.
      AD<double> x1 = vars[x_start + t];
      AD<double> y1 = vars[y_start + t];
      AD<double> psi1 = vars[psi_start + t];
      AD<double> v1 = vars[v_start + t];
      AD<double> cte1 = vars[cte_start + t];
      AD<double> epsi1 = vars[epsi_start + t];

      // The state at time t.
      AD<double> x0 = vars[x_start + t - 1];
      AD<double> y0 = vars[y_start + t - 1];
      AD<double> psi0 = vars[psi_start + t - 1];
      AD<double> v0 = vars[v_start + t - 1];
      AD<double> cte0 = vars[cte_start + t - 1];
      AD<double> epsi0 = vars[epsi_start + t - 1];

      // Look back two timsteps if we can, to account for 100ms one timestep command latency
      int back = 1;
      //if ( t > 1 ) {
      //  back = 2;
      //}

      // Only consider the actuation at time t.
      AD<double> delta0 = vars[delta_start + t - back];
      AD<double> a0 = vars[a_start + t - back];

      AD<double> f0 = coeffs[0] + coeffs[1] * x0 + coeffs[2] * CppAD::pow(x0, 2) + coeffs[3] * CppAD::pow(x0, 3);
      AD<double> psides0 = CppAD::atan(coeffs[1] + 2 * coeffs[2] * x0 + 3 * coeffs[3] * CppAD::pow(x0, 2));

      // The equations for the model:
      // x_[t+1] = x[t] + v[t] * cos(psi[t]) * dt
      // y_[t+1] = y[t] + v[t] * sin(psi[t]) * dt
      // psi_[t+1] = psi[t] + v[t] / Lf * delta[t] * dt
      // v_[t+1] = v[t] + a[t] * dt
      // cte[t+1] = f(x[t]) - y[t] + v[t] * sin(epsi[t]) * dt
      // epsi[t+1] = psi[t] - psides[t] + v[t] * delta[t] / Lf * dt
      fg[1 + x_start    + t] = x1 - (x0 + v0 * CppAD::cos(psi0) * dt);
      fg[1 + y_start    + t] = y1 - (y0 + v0 * CppAD::sin(psi0) * dt);
      fg[1 + psi_start  + t] = psi1 - (psi0 - v0 * delta0 / Lf * dt);
      fg[1 + v_start    + t] = v1 - (v0 + a0 * dt);
      fg[1 + cte_start  + t] = cte1 - ((f0 - y0) + (v0 * CppAD::sin(epsi0) * dt));
      fg[1 + epsi_start + t] = epsi1 - ((psi0 - psides0) - v0 * delta0 / Lf * dt);
    }
  }
};

//
// MPC class definition implementation.
//
MPC::MPC() {}
MPC::~MPC() {}

vector<double> MPC::Solve(Eigen::VectorXd state, Eigen::VectorXd coeffs) {
  bool ok = true;
  typedef CPPAD_TESTVECTOR(double) Dvector;

  // Set the number of model variables (includes both states and inputs).
  size_t n_vars = N * 6 + (N - 1) * 2;

  // Set the number of constraints
  size_t n_constraints = N * 6;

  // Initial value of the independent variables.
  // Should be 0 besides initial values.
  Dvector vars(n_vars);
  for (int i = 0; i < n_vars; i++) {
    vars[i] = 0;
  }

  // Set all non-actuators upper and lowerlimits to the max negative and positive values. They can be anything.
  Dvector vars_lowerbound(n_vars);
  Dvector vars_upperbound(n_vars);
  for (int i = 0; i < delta_start; i++) {
    vars_lowerbound[i] = -1.0e19;
    vars_upperbound[i] = 1.0e19;
  }

  // Hey solver, steering can go from left 25 degrees to right 25 degrees
  for (int i = delta_start; i < a_start; i++) {
    vars_lowerbound[i] = -0.436332; // -25 in rads
    vars_upperbound[i] = 0.436332; // 25 in rads
  }

  // Hey solver, acceleration is ok to go from full backwards to full forwards
  for (int i = a_start; i < n_vars; i++) {
    vars_lowerbound[i] = -1.0;
    vars_upperbound[i] = 1.0;
  }

  // Load state
  double x = state[0];
  double y = state[1];
  double psi = state[2];
  double v = state[3];
  double cte = state[4];
  double epsi = state[5];
  
  // Lower and upper limits for constraints, all of these should be 0 except the initial state indices.
  Dvector constraints_lowerbound(n_constraints);
  Dvector constraints_upperbound(n_constraints);
  for (int i = 0; i < n_constraints; i++) {
    constraints_lowerbound[i] = 0;
    constraints_upperbound[i] = 0;
  }
  constraints_lowerbound[x_start] = x;
  constraints_lowerbound[y_start] = y;
  constraints_lowerbound[psi_start] = psi;
  constraints_lowerbound[v_start] = v;
  constraints_lowerbound[cte_start] = cte;
  constraints_lowerbound[epsi_start] = epsi;
  constraints_upperbound[x_start] = x;
  constraints_upperbound[y_start] = y;
  constraints_upperbound[psi_start] = psi;
  constraints_upperbound[v_start] = v;
  constraints_upperbound[cte_start] = cte;
  constraints_upperbound[epsi_start] = epsi;

  // Set the initial variable values
  vars[x_start] = x;
  vars[y_start] = y;
  vars[psi_start] = psi;
  vars[v_start] = v;
  vars[cte_start] = cte;
  vars[epsi_start] = epsi;

  // Object that computes objective and constraints
  FG_eval fg_eval(coeffs);

  // Options for IPOPT solver
  std::string options;

  // Uncomment this if you'd like more print information
  options += "Integer print_level  0\n";

  // Setting sparse to true allows the solver to take advantage
  // of sparse routines, this makes the computation MUCH FASTER.
  options += "Sparse  true        forward\n";
  options += "Sparse  true        reverse\n";

  // The solver has a maximum time limit of 0.5 seconds.
  options += "Numeric max_cpu_time          0.5\n";

  // The place to return the solution
  CppAD::ipopt::solve_result<Dvector> solution;

  // Solve!
  CppAD::ipopt::solve<Dvector, FG_eval>(
      options, vars, 
      vars_lowerbound, vars_upperbound,
      constraints_lowerbound, constraints_upperbound,
      fg_eval, solution);

  // Check some of the solution values
  ok &= solution.status == CppAD::ipopt::solve_result<Dvector>::success;

  // Cost
//  auto cost = solution.obj_value;
//  std::cout << "Cost " << cost << std::endl;

  // Return actuation
  vector<double> result;
  result.push_back(solution.x[delta_start]);
  result.push_back(solution.x[a_start]);

  // Return x, y co-ords of solution for display
  for (int i = 0; i < N-1; i++) {
    result.push_back(solution.x[x_start + i + 1]);
    result.push_back(solution.x[y_start + i + 1]);
  }
  return result;
}
