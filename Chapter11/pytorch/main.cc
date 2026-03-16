/*
PR's that introduced breaking changes:
  https://github.com/pytorch/pytorch/pull/27422
*/
#include "glovedict.h"
#include "imdbdataset.h"
#include "imdbreader.h"
#include "rnn.h"
#include "vocabulary.h"

#include <experimental/filesystem>
#include <iostream>

namespace fs = std::experimental::filesystem;

float BinaryAccuracy(const torch::Tensor& preds, const torch::Tensor& target) {
  auto rounded_preds = torch::round(torch::sigmoid(preds));
  auto correct =
      torch::eq(rounded_preds, target).to(torch::dtype(torch::kFloat));
  auto acc = correct.sum() / correct.size(0);
  return acc.item<float>();
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> MakeBatchTensors(
    const std::vector<ImdbExample>& batch) {
  // prepare batch data
  std::vector<torch::Tensor> text_data;
  std::vector<torch::Tensor> text_lengths;
  std::vector<torch::Tensor> label_data;
  for (auto& item : batch) {
    text_data.push_back(item.data.first);
    text_lengths.push_back(item.data.second);
    label_data.push_back(item.target);
  }
  // sort items to use them in pack_padded_sequence function
  std::vector<std::size_t> permutation(text_lengths.size());
  std::iota(permutation.begin(), permutation.end(), 0);
  std::sort(permutation.begin(), permutation.end(),
            [&](std::size_t i, std::size_t j) {
              return text_lengths[i].item().toLong() <
                     text_lengths[j].item().toLong();
            });
  std::reverse(permutation.begin(),
               permutation.end());  // we need decreasing order
  auto appy_permutation = [&permutation](
                              const std::vector<torch::Tensor>& vec) {
    std::vector<torch::Tensor> sorted_vec(vec.size());
    std::transform(permutation.begin(), permutation.end(), sorted_vec.begin(),
                   [&](std::size_t i) { return vec[i]; });
    return sorted_vec;
  };
  text_data = appy_permutation(text_data);
  text_lengths = appy_permutation(text_lengths);
  label_data = appy_permutation(label_data);

  // stack batch items to single tensors
  torch::Tensor texts = torch::stack(text_data);
  torch::Tensor lengths = torch::stack(text_lengths);
  torch::Tensor labels = torch::stack(label_data);

  return {texts, lengths, labels};
}

void TrainModel(
    int epoch,
    SentimentRNN& model,
    torch::optim::Optimizer& optimizer,
    torch::data::StatelessDataLoader<ImdbDataset,
                                     torch::data::samplers::RandomSampler>&
        train_loader) {
  double epoch_loss = 0;
  double epoch_acc = 0;
  model->train();  // switch to the training mode
  // Iterate the data loader to get batches from the dataset
  int batch_index = 0;
  for (auto& batch : train_loader) {
    // Clear gradients
    optimizer.zero_grad();

    // prepare batch data
    torch::Tensor texts, lengths, labels;
    std::tie(texts, lengths, labels) = MakeBatchTensors(batch);

    /* Execute the model on the input data
     * transpose it to match the rnn input shape
     * [seq_len, batch_size, features]
     */
    torch::Tensor prediction = model->forward(texts.t(), lengths);
    prediction.squeeze_(1);

    // test data
    // std::cout << prediction << std::endl;
    // std::cout << labels << std::endl;

    // Compute a loss value to estimate error of our model
    // target should have size of [batch_size]
    torch::Tensor loss = torch::binary_cross_entropy_with_logits(
        prediction, labels, {}, {}, torch::Reduction::Mean);

    // Compute gradients of the loss and parameters of our model
    loss.backward();

    // Update the parameters based on the calculated gradients.
    optimizer.step();

    auto loss_value = static_cast<double>(loss.item<float>());
    auto acc_value = static_cast<double>(BinaryAccuracy(prediction, labels));
    epoch_loss += loss_value;
    epoch_acc += acc_value;

    // Output the loss every 10 batches.
    if (++batch_index % 10 == 0) {
      std::cout << "Epoch: " << epoch << " | Batch: " << batch_index
                << " | Loss: " << loss_value << " | Acc: " << acc_value
                << std::endl;
    }
  }

  std::cout << "Epoch: " << epoch
            << " | Loss: " << (epoch_loss / (batch_index - 1))
            << " | Acc: " << (epoch_acc / (batch_index - 1)) << std::endl;
}

void TestModel(
    int epoch,
    SentimentRNN& model,
    torch::data::StatelessDataLoader<ImdbDataset,
                                     torch::data::samplers::RandomSampler>&
        test_loader) {
  torch::NoGradGuard guard;
  double epoch_loss = 0;
  double epoch_acc = 0;
  model->eval();  // switch to the evaluation mode
  // Iterate the data loader to get batches from the dataset
  int batch_index = 0;
  for (auto& batch : test_loader) {
    // prepare batch data
    torch::Tensor texts, lengths, labels;
    std::tie(texts, lengths, labels) = MakeBatchTensors(batch);

    /* Execute the model on the input data
     * transpose it to match the rnn input shape
     * [seq_len, batch_size, features]
     */
    torch::Tensor prediction = model->forward(texts.t(), lengths);
    prediction.squeeze_(1);

    // Compute a loss value to estimate error of our model
    // target should have size of [batch_size]
    torch::Tensor loss = torch::binary_cross_entropy_with_logits(
        prediction, labels, {}, {}, torch::Reduction::Mean);

    auto loss_value = static_cast<double>(loss.item<float>());
    auto acc_value = static_cast<double>(BinaryAccuracy(prediction, labels));
    epoch_loss += loss_value;
    epoch_acc += acc_value;

    ++batch_index;
  }

  std::cout << "Epoch: " << epoch
            << " | Test Loss: " << (epoch_loss / (batch_index - 1))
            << " | Test Acc: " << (epoch_acc / (batch_index - 1)) << std::endl;
}

int main(int argc, char** argv) {
  torch::DeviceType device = torch::cuda::is_available()
                                 ? torch::DeviceType::CUDA
                                 : torch::DeviceType::CPU;

  if (argc > 1) {
    try {
      auto root_path = fs::path(argv[1]);
      if (!fs::exists(root_path))
        throw std::invalid_argument("dataset folder missing");
      auto glove_path = fs::path(argv[2]);
      if (!fs::exists(root_path))
        throw std::invalid_argument("Glove file missing");

      ImdbReader train_reader(root_path / "train");
      ImdbReader test_reader(root_path / "test");

      WordsFrequencies words_frequencies;
      GetWordsFrequencies(train_reader, words_frequencies);
      GetWordsFrequencies(test_reader, words_frequencies);

      int64_t vocab_size = 25000;
      SelectTopFrequencies(words_frequencies, vocab_size);

      int64_t embedding_dim = 100;
      GloveDict glove_dict(glove_path, embedding_dim);

      Vocabulary vocab(words_frequencies, glove_dict);

      // create datasets
      ImdbDataset train_dataset(&train_reader, &vocab, device);
      ImdbDataset test_dataset(&test_reader, &vocab, device);

      // init data loaders
      size_t batch_size = 32;
      auto train_loader = torch::data::make_data_loader(
          train_dataset,
          torch::data::DataLoaderOptions().batch_size(batch_size).workers(4));

      auto test_loader = torch::data::make_data_loader(
          test_dataset,
          torch::data::DataLoaderOptions().batch_size(batch_size).workers(4));

      // initialize model
      int64_t hidden_dim = 256;
      int64_t output_dim = 1;
      int64_t n_layers = 2;
      bool bidirectional = true;
      double dropout = 0.5;
      int64_t pad_idx = vocab.GetPaddingIndex();

      SentimentRNN model(vocab.GetEmbeddingsCount(), embedding_dim, hidden_dim,
                         output_dim, n_layers, bidirectional, dropout, pad_idx);

      model->SetPretrainedEmbeddings(vocab.GetEmbeddings());

      // initilize optimizer
      double learning_rate = 0.01;
      torch::optim::Adam optimizer(model->parameters(),
                                   torch::optim::AdamOptions(learning_rate));

      // training
      model->to(device);
      int epochs = 100;
      for (int epoch = 0; epoch < epochs; ++epoch) {
        TrainModel(epoch, model, optimizer, *train_loader);
        TestModel(epoch, model, *test_loader);
      }

    } catch (const std::exception& err) {
      std::cerr << err.what() << std::endl;
    }
  } else {
    std::cerr
        << "Please specify a path to the dataset and glove vectors file\n";
  }
  return 1;
}
