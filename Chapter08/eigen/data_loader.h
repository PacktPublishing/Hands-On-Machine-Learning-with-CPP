#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <fstream>
#include <map>
#include <sstream>
#include <unordered_map>
#include <vector>

using Movies = std::map<int32_t, std::string>;

using UserRate = std::pair<int32_t, float>;
using Ratings = std::unordered_map<int32_t, std::vector<UserRate>>;

Movies LoadMovies(const std::string& path) {
  Movies movies;
  int32_t row = 0;
  std::ifstream indata(path);
  if (indata) {
    std::string line;

    while (std::getline(indata, line)) {
      if (row != 0) {
        std::stringstream line_stream(line);
        std::string cell;
        std::getline(line_stream, cell, ',');
        auto id = std::stoi(cell);
        std::getline(line_stream, cell, ',');
        movies.insert({id, cell});
      }
      ++row;
    }
  }
  return movies;
}

Ratings LoadRatings(const std::string& path) {
  Ratings ratings;
  std::ifstream indata(path);
  if (indata) {
    std::string line;
    int32_t row = 0;
    while (std::getline(indata, line)) {
      if (row != 0) {
        std::stringstream line_stream(line);
        std::string cell;
        std::getline(line_stream, cell, ',');
        auto user_id = std::stoi(cell);
        std::getline(line_stream, cell, ',');
        auto movie_id = std::stoi(cell);
        std::getline(line_stream, cell, ',');
        auto rating = std::stof(cell);
        ratings[user_id].push_back({movie_id, rating});
      }
      ++row;
    }
  }
  return ratings;
}

#endif  // DATA_LOADER_H
