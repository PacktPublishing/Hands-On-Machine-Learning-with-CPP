#include <torch/torch.h>

#include <iostream>
#include <random>

float func(float x) {
  return 4.f + 0.3f * x;
}

class NetImpl : public torch::nn::Module {
 public:
  NetImpl() {
    l1_ = torch::nn::Linear(torch::nn::LinearOptions(1, 8).with_bias(true));
    register_module("l1", l1_);
    l2_ = torch::nn::Linear(torch::nn::LinearOptions(8, 4).with_bias(true));
    register_module("l2", l2_);
    l3_ = torch::nn::Linear(torch::nn::LinearOptions(4, 1).with_bias(true));
    register_module("l3", l3_);

    // initialize weights
    for (auto m : modules(false)) {
      if (m->name().find("Linear") != std::string::npos) {
        for (auto& p : m->named_parameters()) {
          if (p.key().find("weight") != std::string::npos) {
            torch::nn::init::normal_(p.value(), 0, 0.01);
          }
          if (p.key().find("bias") != std::string::npos) {
            torch::nn::init::zeros_(p.value());
          }
        }
      }
    }
  }

  torch::Tensor forward(torch::Tensor x) {
    auto y = l1_(x);
    y = l2_(y);
    y = l3_(y);
    return y;
  }

  void SaveWeights(const std::string& file_name) {
    torch::serialize::OutputArchive archive;
    auto parameters = named_parameters(true /*recurse*/);
    auto buffers = named_buffers(true /*recurse*/);
    for (const auto& param : parameters) {
      if (param.value().defined()) {
        archive.write(param.key(), param.value());
      }
    }
    for (const auto& buffer : buffers) {
      if (buffer.value().defined()) {
        archive.write(buffer.key(), buffer.value(), /*is_buffer*/ true);
      }
    }
    archive.save_to(file_name);
  }

  void LoadWeights(const std::string& file_name) {
    torch::serialize::InputArchive archive;
    archive.load_from(file_name);
    torch::NoGradGuard no_grad;
    auto parameters = named_parameters(true /*recurse*/);
    auto buffers = named_buffers(true /*recurse*/);
    for (auto& param : parameters) {
      archive.read(param.key(), param.value());
    }
    for (auto& buffer : buffers) {
      archive.read(buffer.key(), buffer.value(), /*is_buffer*/ true);
    }
  }

 private:
  torch::nn::Linear l1_{nullptr};
  torch::nn::Linear l2_{nullptr};
  torch::nn::Linear l3_{nullptr};
};

TORCH_MODULE(Net);

int main() {
  try {
    torch::DeviceType device = torch::cuda::is_available()
                                   ? torch::DeviceType::CUDA
                                   : torch::DeviceType::CPU;

    std::random_device rd;
    std::mt19937 re(rd());
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);

    // generate data
    size_t n = 1000;
    torch::Tensor x;
    torch::Tensor y;
    {
      std::vector<float> values(n);
      std::iota(values.begin(), values.end(), 0);
      std::shuffle(values.begin(), values.end(), re);

      std::vector<torch::Tensor> x_vec(n);
      std::vector<torch::Tensor> y_vec(n);
      for (size_t i = 0; i < n; ++i) {
        x_vec[i] = torch::tensor(
            values[i],
            torch::dtype(torch::kFloat).device(device).requires_grad(false));

        y_vec[i] = torch::tensor(
            (func(values[i]) + dist(re)),
            torch::dtype(torch::kFloat).device(device).requires_grad(false));
      }
      x = torch::stack(x_vec);
      y = torch::stack(y_vec);
    }

    // normalize data
    auto x_mean = torch::mean(x, /*dim*/ 0);
    auto x_std = torch::std(x, /*dim*/ 0);
    x = (x - x_mean) / x_std;

    Net model;
    model->to(device);

    // initilize optimizer ----------------------------------------------
    double learning_rate = 0.01;
    torch::optim::Adam optimizer(
        model->parameters(),
        torch::optim::AdamOptions(learning_rate).weight_decay(0.00001));

    // training
    int64_t batch_size = 10;
    int64_t batches_num = static_cast<int64_t>(n) / batch_size;
    int epochs = 10;
    for (int epoch = 0; epoch < epochs; ++epoch) {
      // train the model -----------------------------------------------
      model->train();  // switch to the training mode

      // Iterate the data
      double epoch_loss = 0;
      for (int64_t batch_index = 0; batch_index < batches_num; ++batch_index) {
        auto batch_x = x.narrow(0, batch_index * batch_size, batch_size);
        auto batch_y = y.narrow(0, batch_index * batch_size, batch_size);

        // Clear gradients
        optimizer.zero_grad();

        // Execute the model on the input data
        torch::Tensor prediction = model->forward(batch_x);

        torch::Tensor loss = torch::mse_loss(prediction, batch_y);

        // Compute gradients of the loss and parameters of our model
        loss.backward();

        // Update the parameters based on the calculated gradients.
        optimizer.step();

        epoch_loss += static_cast<double>(loss.item<float>());
      }
      std::cout << "Epoch: " << epoch << " | Loss: " << epoch_loss / batches_num
                << std::endl;
    }
    // serialize model
    // model->SaveWeights("pytorch_net.dat");

    torch::save(model, "pytorch_net.pt");

    // Test

    Net model_loaded;
    torch::load(model_loaded, "pytorch_net.pt");
    // model_loaded->LoadWeights("pytorch_net.dat");

    model_loaded->eval();
    std::cout << "Test:\n";
    for (int i = 0; i < 5; ++i) {
      auto x_val = static_cast<float>(i) + 0.1f;
      auto tx =
          torch::tensor(x_val, torch::dtype(torch::kFloat).device(device));
      tx = (tx - x_mean) / x_std;

      auto ty = torch::tensor(func(x_val),
                              torch::dtype(torch::kFloat).device(device));

      torch::Tensor prediction = model_loaded->forward(tx);

      std::cout << "Target:" << ty << std::endl;
      std::cout << "Prediction:" << prediction << std::endl;
    }

    return 0;
  } catch (const std::exception& err) {
    std::cerr << err.what();
  }
  return 1;
}
