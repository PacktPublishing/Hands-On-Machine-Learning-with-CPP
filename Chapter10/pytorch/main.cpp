#include "lenet5.h"
#include "mnistdataset.h"

#include <experimental/filesystem>
#include <iostream>

namespace fs = std::experimental::filesystem;

int main(int argc, char** argv) {
  if (argc > 1) {
    auto root_path = fs::path(argv[1]);
    if (fs::exists(root_path)) {
      auto train_images = root_path / "train-images-idx3-ubyte";
      auto train_labels = root_path / "train-labels-idx1-ubyte";
      auto test_images = root_path / "t10k-images-idx3-ubyte";
      auto test_labels = root_path / "t10k-labels-idx1-ubyte";
      if (fs::exists(train_images) && fs::exists(train_labels) &&
          fs::exists(test_images) && fs::exists(test_labels)) {
        try {
          torch::DeviceType device = torch::cuda::is_available()
                                         ? torch::DeviceType::CUDA
                                         : torch::DeviceType::CPU;

          
          // initialize train dataset
          // ----------------------------------------------
          MNISTDataset train_dataset(train_images.native(),
                                     train_labels.native(),
                                     device);

          // test that loaded data is correct
          // train_dataset.ShowItem(4678);

          auto train_loader = torch::data::make_data_loader(
              train_dataset.map(torch::data::transforms::Stack<>()),
              torch::data::DataLoaderOptions().batch_size(256).workers(
                  8));  // random sampler is default

          // initialize test dataset
          // ----------------------------------------------
          MNISTDataset test_dataset(test_images.native(), 
                                    test_labels.native(),
                                    device);

          auto test_loader = torch::data::make_data_loader(
              test_dataset.map(torch::data::transforms::Stack<>()),
              torch::data::DataLoaderOptions().batch_size(1024).workers(
                  8));  // random sampler is default

          // initialize net
          LeNet5 model;
          model->to(device);

          // initilize optimizer ----------------------------------------------
          double learning_rate = 0.01;
          double weight_decay = 0.0001;  // regularization parameter
          torch::optim::SGD optimizer(model->parameters(),
                                      torch::optim::SGDOptions(learning_rate)
                                          .weight_decay(weight_decay)
                                          .momentum(0.5));

          // training
          int epochs = 100;
          for (int epoch = 0; epoch < epochs; ++epoch) {
            // train the model -----------------------------------------------
            model->train();  // switch to the training mode

            // Iterate the data loader to get batches from the dataset
            int batch_index = 0;
            for (auto& batch : (*train_loader)) {
              // Clear gradients
              optimizer.zero_grad();

              // Execute the model on the input data
              torch::Tensor prediction = model->forward(batch.data);

              // test data
              // std::cout << prediction << std::endl;
              // std::cout << batch.target << std::endl;

              // Compute a loss value to estimate error of our model
              // target should have size of [batch_size]
              torch::Tensor loss =
                  torch::nll_loss(prediction, batch.target.squeeze(1));

              // Compute gradients of the loss and parameters of our model
              loss.backward();

              // Update the parameters based on the calculated gradients.
              optimizer.step();

              // Output the loss every 10 batches.
              if (++batch_index % 10 == 0) {
                std::cout << "Epoch: " << epoch << " | Batch: " << batch_index
                          << " | Loss: " << loss.item<float>() << std::endl;
              }
            }

            // evaluate model on the test data -------------------------------
            model->eval();  // switch to the training mode
            unsigned long total_correct = 0;
            float avg_loss = 0.0;
            for (auto& batch : (*test_loader)) {
              // Execute the model on the input data
              torch::Tensor prediction = model->forward(batch.data);

              // Compute a loss value to estimate error of our model
              torch::Tensor loss =
                  torch::nll_loss(prediction, batch.target.squeeze(1));

              avg_loss += loss.sum().item<float>();
              // max - returns a tuple (values, indices) where values is
              // the maximum value of each row of the input tensor in the given
              // dimension dim. And indices is the index location of each
              // maximum value found (argmax).
              auto pred = std::get<1>(prediction.detach_().max(1));
              total_correct += static_cast<unsigned long>(
                  pred.eq(batch.target.view_as(pred)).sum().item<long>());
            }
            avg_loss /= test_dataset.size().value();
            double accuracy = (static_cast<double>(total_correct) /
                               test_dataset.size().value());
            std::cout << "Test Avg. Loss: " << avg_loss
                      << " | Accuracy: " << accuracy << std::endl;
          }

          return 0;
        } catch (const std::exception& err) {
          std::cerr << err.what();
          return 1;
        }
      }
    }
  }
  std::cerr << "Please specify correct dataset folder\n";
  return 1;
}
