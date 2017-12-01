#include <math.h>
#include <uWS/uWS.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "MPC.h"
#include "json.hpp"

#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/cc/ops/image_ops.h>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/lib/io/path.h>

using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::Session;
using tensorflow::string;
using tensorflow::int32;

static Status ReadEntireFile(tensorflow::Env* env, const string& filename, Tensor* output) {
  tensorflow::uint64 file_size = 0;
  TF_RETURN_IF_ERROR(env->GetFileSize(filename, &file_size));

  string contents;
  contents.resize(file_size);

  std::unique_ptr<tensorflow::RandomAccessFile> file;
  TF_RETURN_IF_ERROR(env->NewRandomAccessFile(filename, &file));

  tensorflow::StringPiece data;
  TF_RETURN_IF_ERROR(file->Read(0, file_size, &data, &(contents)[0]));
  if (data.size() != file_size) {
    return tensorflow::errors::DataLoss("Truncated read of '", filename,
                                        "' expected ", file_size, " got ",
                                        data.size());
  }
  output->scalar<string>()() = data.ToString();
  return Status::OK();
}

// Given an image, try to decode it as an image, resize it to the requested size, and then scale the values as desired.
Status ReadTensorFromImageString(const string& image, const int input_height, const int input_width, 
                                 const float input_mean, const float input_std,
                                 std::vector<Tensor>* out_tensors) {
  auto root = tensorflow::Scope::NewRootScope();
  using namespace ::tensorflow::ops;

  // Copy into a tensor named input
  Tensor input(tensorflow::DT_STRING, tensorflow::TensorShape());
  input.scalar<string>()() = image;

  // Use a placeholder to read input data
  auto input_string_placeholder = Placeholder(root.WithOpName("input"), tensorflow::DataType::DT_STRING);

  // Create input Tensors (just the one)
  std::vector<std::pair<string, tensorflow::Tensor>> inputs = { {"input", input}, };

  // Decode it
  tensorflow::Output image_reader = DecodeJpeg(root.WithOpName("jpeg_reader"), input_string_placeholder, DecodeJpeg::Channels(3));

  // Now cast the image data to float so we can do normal math on it.
  auto float_caster = Cast(root.WithOpName("float_caster"), image_reader, tensorflow::DT_FLOAT);

  // The convention for image ops in TensorFlow is that all images are expected
  // to be in batches, so that they're four-dimensional arrays with indices of
  // [batch, height, width, channel]. Because we only have a single image, we
  // have to add a batch dimension of 1 to the start with ExpandDims().
  auto dims_expander = ExpandDims(root, float_caster, 0);

  // Bilinearly resize the image to fit the required dimensions
  auto resized = ResizeBilinear(root, dims_expander, Const(root.WithOpName("size"), {input_height, input_width}));

  // Subtract the mean and divide by the scale
  string output_name = "normalized";
  Div(root.WithOpName(output_name), Sub(root, resized, {input_mean}), {input_std});

  // This runs the GraphDef network definition that we've just constructed, and returns the results in the output tensor
  tensorflow::GraphDef graph;
  TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));
  std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));
  TF_RETURN_IF_ERROR(session->Create(graph));
  TF_RETURN_IF_ERROR(session->Run({inputs}, {output_name}, {}, out_tensors));
  return Status::OK();
}


// Given an image file name, read in the data, try to decode it as an image,
// resize it to the requested size, and then scale the values as desired.
Status ReadTensorFromImageFile(const string& file_name, const int input_height,
                               const int input_width, const float input_mean,
                               const float input_std,
                               std::vector<Tensor>* out_tensors) {
  auto root = tensorflow::Scope::NewRootScope();
  using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

  string output_name = "normalized";

  // read file_name into a tensor named input
  Tensor input(tensorflow::DT_STRING, tensorflow::TensorShape());
  TF_RETURN_IF_ERROR(ReadEntireFile(tensorflow::Env::Default(), file_name, &input));

  // use a placeholder to read input data
  auto file_reader = Placeholder(root.WithOpName("input"), tensorflow::DataType::DT_STRING);

  std::vector<std::pair<string, tensorflow::Tensor>> inputs = { {"input", input}, };

  // Now try to figure out what kind of file it is and decode it.
  const int wanted_channels = 3;
  tensorflow::Output image_reader;
  if (tensorflow::StringPiece(file_name).ends_with(".png")) {
    image_reader = DecodePng(root.WithOpName("png_reader"), file_reader, DecodePng::Channels(wanted_channels));
  } else {
    // Assume if it's neither a PNG nor a GIF then it must be a JPEG.
    image_reader = DecodeJpeg(root.WithOpName("jpeg_reader"), file_reader, DecodeJpeg::Channels(wanted_channels));
  }

  // Now cast the image data to float so we can do normal math on it.
  auto float_caster = Cast(root.WithOpName("float_caster"), image_reader, tensorflow::DT_FLOAT);

  // The convention for image ops in TensorFlow is that all images are expected
  // to be in batches, so that they're four-dimensional arrays with indices of
  // [batch, height, width, channel]. Because we only have a single image, we
  // have to add a batch dimension of 1 to the start with ExpandDims().
  auto dims_expander = ExpandDims(root, float_caster, 0);

  // Bilinearly resize the image to fit the required dimensions.
  auto resized = ResizeBilinear( root, dims_expander, Const(root.WithOpName("size"), {input_height, input_width}));

  // Subtract the mean and divide by the scale.
  Div(root.WithOpName(output_name), Sub(root, resized, {input_mean}), {input_std});

  // This runs the GraphDef network definition that we've just constructed, and
  // returns the results in the output tensor.
  tensorflow::GraphDef graph;
  TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

  std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));
  TF_RETURN_IF_ERROR(session->Create(graph));
  TF_RETURN_IF_ERROR(session->Run({inputs}, {output_name}, {}, out_tensors));
  return Status::OK();
}

// Reads a model graph definition from disk, and creates a session object you
// can use to run it.
Status LoadGraph(const string& graph_file_name, std::unique_ptr<tensorflow::Session>* session) {
  tensorflow::GraphDef graph_def;
  Status load_graph_status = ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
  if (!load_graph_status.ok()) {
    return tensorflow::errors::NotFound("Failed to load compute graph at '", graph_file_name, "'");
  }
  session->reset(tensorflow::NewSession(tensorflow::SessionOptions()));
  Status session_create_status = (*session)->Create(graph_def);
  if (!session_create_status.ok()) {
    return session_create_status;
  }
  return Status::OK();
}

std::vector<Tensor> run_image(Session* session, const string& jpg) {

  int32 input_width = 320;
  int32 input_height = 160;
  float input_mean = 0;
  float input_std = 255;

  // Set up outputs
  std::vector<Tensor> resized_tensors;

/*  string image_path = tensorflow::io::JoinPath("", filename);
  Status read_tensor_status = ReadTensorFromImageFile(image_path, input_height, input_width, input_mean, input_std, &resized_tensors);
  if (!read_tensor_status.ok()) {
    LOG(ERROR) << read_tensor_status;
  }*/
  Status status = ReadTensorFromImageString(jpg, input_height, input_width, input_mean, input_std, &resized_tensors);
  if (!status.ok()) {
    LOG(ERROR) << status;
  }
  const Tensor& resized_tensor = resized_tensors[0];

  // Actually run the image through the model.
  string input_layer = "main_input";
  string output_layer0 = "output_0";
  string output_layer1 = "output_1";
  std::vector<Tensor> outputs;
  status = session->Run({{input_layer, resized_tensor}}, {output_layer0, output_layer1}, {}, &outputs);
  if (!status.ok()) {
    LOG(ERROR) << status;
  }

  return outputs;
}


// for convenience
using json = nlohmann::json;

// For converting back and forth between radians and degrees.
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
string hasData(string s) {
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.rfind("}]");
  if (found_null != string::npos) {
    return "";
  } else if (b1 != string::npos && b2 != string::npos) {
    return s.substr(b1, b2 - b1 + 2);
  }
  return "";
}

// Evaluate a polynomial.
double polyeval(Eigen::VectorXd coeffs, double x) {
  double result = 0.0;
  for (int i = 0; i < coeffs.size(); i++) {
    result += coeffs[i] * pow(x, i);
  }
  return result;
}

// Fit a polynomial.
// Adapted from
// https://github.com/JuliaMath/Polynomials.jl/blob/master/src/Polynomials.jl#L676-L716
Eigen::VectorXd polyfit(Eigen::VectorXd xvals, Eigen::VectorXd yvals, int order) {
  assert(xvals.size() == yvals.size());
  assert(order >= 1 && order <= xvals.size() - 1);
  Eigen::MatrixXd A(xvals.size(), order + 1);

  for (int i = 0; i < xvals.size(); i++) {
    A(i, 0) = 1.0;
  }

  for (int j = 0; j < xvals.size(); j++) {
    for (int i = 0; i < order; i++) {
      A(j, i + 1) = A(j, i) * xvals(j);
    }
  }

  auto Q = A.householderQr();
  auto result = Q.solve(yvals);
  return result;
}

static std::string base64_decode(const std::string &in) {

    std::string out;

    std::vector<int> T(256,-1);
    for (int i=0; i<64; i++) T["ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"[i]] = i; 

    int val=0, valb=-8;
    for (unsigned char c : in) {
        if (T[c] == -1) break;
        val = (val<<6) + T[c];
        valb += 6;
        if (valb>=0) {
            out.push_back(char((val>>valb)&0xFF));
            valb-=8;
        }
    }
    return out;
}

int main() {
  uWS::Hub h;

  MPC mpc;

  using namespace tensorflow;
  using namespace tensorflow::ops;

  // Initialize a tensorflow session
  Session* session;
  Status status = NewSession(SessionOptions(), &session);
  if (!status.ok()) {
    std::cout << status.ToString() << "\n";
    return 1;
  }

  const string& graph_file_name = "graph.pb";
  tensorflow::GraphDef graph_def;
  Status load_graph_status = ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
  if (!load_graph_status.ok()) {
    printf("Failed to load compute graph");
  }

  // Add the graph to the session
  status = session->Create(graph_def);
  if (!status.ok()) {
    std::cout << status.ToString() << "\n";
    return 1;
  }


  h.onMessage([&mpc, &session](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
                     uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    string sdata = string(data).substr(0, length);
//    cout << sdata << endl;
    if (sdata.size() > 2 && sdata[0] == '4' && sdata[1] == '2') {
      string s = hasData(sdata);
      if (s != "") {
        auto j = json::parse(s);
        string event = j[0].get<string>();
        if (event == "telemetry") {
          // j[1] is the data JSON object
          vector<double> ptsx = j[1]["ptsx"];
          vector<double> ptsy = j[1]["ptsy"];
          double px = j[1]["x"];
          double py = j[1]["y"];
          double psi = j[1]["psi"];
          double v = j[1]["speed"];

          // Get image
          const string imgString = base64_decode(j[1]["image"]);

          // Run the image
          std::vector<Tensor> outputs = run_image(session, imgString);

          // Print output
          std::cout << outputs[1].flat<float>()(3) << "\n";

          float wp = outputs[1].flat<float>()(3) * 3; // * 3 for scaling

          // Adjust waypoints
          ptsx[0] -= 2 * wp;
          ptsx[1] -= 4 * wp;
          ptsx[2] -= 6 * wp;
          ptsx[3] -= 6 * wp;
          ptsx[4] -= 4 * wp;
          ptsx[5] -= 2 * wp;

          // Transform
          vector<double> ptsx_transformed;
          vector<double> ptsy_transformed;
          for (int i = 0; i < ptsx.size(); i++) {
            double delta_x = ptsx[i] - px;
            double delta_y = ptsy[i] - py;
            ptsx_transformed.push_back(delta_x * cos(-psi) - delta_y * sin(-psi));
            ptsy_transformed.push_back(delta_x * sin(-psi) + delta_y * cos(-psi));
          }

          // Make into an eigen
          double* pointer_x = &ptsx_transformed[0];
          double* pointer_y = &ptsy_transformed[0];
          Eigen::Map<Eigen::VectorXd> ptsx_transformed_eigen(pointer_x, 6);
          Eigen::Map<Eigen::VectorXd> ptsy_transformed_eigen(pointer_y, 6);

          // Fit a polynomial to the above x and y coordinates
          auto coeffs = polyfit(ptsx_transformed_eigen, ptsy_transformed_eigen, 3);

          // The cross track error is calculated by evaluating a polynomial at x (0), f(x) and subtracting y (0).
          double cte = polyeval(coeffs, 0);

          // Due to the sign starting at 0, the orientation error is -f'(x).
          double epsi = psi - atan(coeffs[1]);

          // Set state
          Eigen::VectorXd state(6);
          state << 0, 0, 0, v, cte, epsi;

          // Calculate steering angle and throttle using MPC.
          // Both are in between [-1, 1].
          auto actuation = mpc.Solve(state, coeffs);

          // Don't make me use the CLAMPS!
          // Divide by deg2rad(25) before you send the steering value back.
          double steer_value = fmin(fmax(actuation[0]/deg2rad(25), -1.0), 1.0);
          double throttle_value = fmin(fmax(actuation[1], -1.0), 1.0);

          // Debug
          //std::cout << "Throttle: " << throttle_value << " Steering Value: " << steer_value << std::endl;

          // Put in message
          json msgJson;
          msgJson["steering_angle"] = steer_value;
          msgJson["throttle"] = throttle_value;

          // Display the MPC predicted trajectory
          // Add the (x,y) points. Points are in reference to the vehicle's coordinate system.
          // The points in the simulator are shown by a green line.
          vector<double> mpc_x_vals;
          vector<double> mpc_y_vals;
          for (int i = 2; i < actuation.size(); i += 2) {
            mpc_x_vals.push_back(actuation[i]);
            mpc_y_vals.push_back(actuation[i+1]);
          }
          msgJson["mpc_x"] = mpc_x_vals;
          msgJson["mpc_y"] = mpc_y_vals;

          // Add the (x,y) points. Points are in reference to the vehicle's coordinate system.
          // The points in the simulator are shown by a yellow line.
          vector<double> next_x_vals;
          vector<double> next_y_vals;
          for (int i = 0; i < 100; i += 5){
            next_x_vals.push_back(i);
            next_y_vals.push_back(polyeval(coeffs, i));
          }
          msgJson["next_x"] = next_x_vals;
          msgJson["next_y"] = next_y_vals;

          // Create the message
          auto msg = "42[\"steer\"," + msgJson.dump() + "]";
          //std::cout << msg << std::endl;

          // Latency
          // The purpose is to mimic real driving conditions where
          // the car does actuate the commands instantly.
          //
          // Feel free to play around with this value but should be to drive
          // around the track with 100ms latency.
          //
          // NOTE: REMEMBER TO SET THIS TO 100 MILLISECONDS BEFORE SUBMITTING.
          this_thread::sleep_for(chrono::milliseconds(100));
          ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        }
      } else {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }
  });

  // We don't need this since we're not using HTTP but if it's removed the
  // program doesn't compile :-(
  h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data,
                     size_t, size_t) {
    const std::string s = "<h1>Hello world!</h1>";
    if (req.getUrl().valueLength == 1) {
      res->end(s.data(), s.length());
    } else {
      // i guess this should be done more gracefully?
      res->end(nullptr, 0);
    }
  });

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
                         char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port)) {
    std::cout << "Listening to port " << port << std::endl;
  } else {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  h.run();

  // Free any resources used by the session
//  status = session->Close();

  return 0;
}
