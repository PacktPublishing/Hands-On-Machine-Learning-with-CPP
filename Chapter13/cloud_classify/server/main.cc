#include <torch/script.h>
#include "network.h"
#include <httplib/httplib.h>
#include "utils.h"

int main(int argc, char** argv) {
  try {
    std::string snapshoot_path;
    std::string synset_path;
    std::string www_path;
    std::string host = "localhost";
    int port = 8080;

    if (argc >= 4) {
      snapshoot_path = argv[1];
      synset_path = argv[2];
      www_path = argv[3];
      if (argc >= 5)
        host = argv[4];
      if (argc >= 6)
        port = std::stoi(argv[5]);

      std::cout << "Starting server:\n";
      std::cout << "Model path: " << snapshoot_path << "\n";
      std::cout << "Synset path: " << synset_path << "\n";
      std::cout << "WWW path: " << www_path << "\n";
      std::cout << "Host: " << host << "\n";
      std::cout << "Port: " << port << std::endl;

      torch::DeviceType device_type = torch::cuda::is_available()
                                          ? torch::DeviceType::CUDA
                                          : torch::DeviceType::CPU;

      Network network(snapshoot_path, synset_path, device_type);

      std::cout << "Model initialized" << std::endl;

      httplib::Server server;
      server.set_base_dir(www_path.c_str());
      server.set_logger([](const auto& req, const auto& /*res*/) {
        std::cout << req.method << "\n" << req.path << std::endl;
      });

      server.set_error_handler([](const auto& /*req*/, auto& res) {
        std::stringstream buf;
        buf << "<p>Error Status: <span style='color:red;'>";
        buf << res.status;
        buf << "</span></p>";
        res.set_content(buf.str(), "text/html");
      });

      server.Post("/multipart", [&](const auto& req, auto& res) {
        std::string response_string;
        for (auto& file : req.files) {
          auto body = file.second.content;
          std::cout << file.second.filename << std::endl;
          std::cout << file.second.content_type << std::endl;
          try {
            auto img = ReadMemoryImageTensor(body, 224, 224);
            auto class_name = network.Classify(img);
            response_string += "; " + class_name;
          } catch (...) {
            response_string += "; Classification failed";
          }
        }
        res.set_content(response_string.c_str(), "text/html");
      });

      std::cout << "Server starting ..." << std::endl;

      if (!server.listen(host.c_str(), port)) {
        std::cerr << "Failed to start server\n";
      }

      std::cout << "Server finished" << std::endl;
    } else {
      std::cout << "usage: " << argv[0]
                << " <model snapshoot path> <synset file path> <www "
                   "dir=../../client> "
                   "[host=localhost] [port=8080]\n";
    }
  } catch (const std::exception& err) {
    std::cerr << err.what();
  } catch (...) {
    std::cerr << "Unhandled exception";
  }
  return 1;
}
