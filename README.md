# Hands-On-Machine-Learning-with-C++

<a href="https://www.packtpub.com/data/hands-on-machine-learning-with-c?utm_source=github&utm_medium=repository&utm_campaign=9781789955330"><img src="https://www.packtpub.com/media/catalog/product/cache/e4d64343b1bc593f1c5348fe05efa4a6/9/7/9781789955330-original.png" alt="Hands-On Machine Learning with C++" height="256px" align="right"></a>

This is the code repository for [Hands-On Machine Learning with C++](https://www.packtpub.com/data/hands-on-machine-learning-with-c?utm_source=github&utm_medium=repository&utm_campaign=9781789955330), published by Packt.

**Build, train, and deploy end-to-end machine learning and deep learning pipelines**

## What is this book about?
This book will help you explore how to implement different well-known machine learning algorithms with various C++ frameworks and libraries. You will cover basic to advanced machine learning concepts with practical and easy to follow examples. By the end of the book, you will be able to build various machine learning models with ease.

This book covers the following exciting features: 
* Explore how to load and preprocess various data types to suitable C++ data structures
* Employ key machine learning algorithms with various C++ libraries
* Understand the grid-search approach to find the best parameters for a machine learning model
* Implement an algorithm for filtering anomalies in user data using Gaussian distribution
* Improve collaborative filtering to deal with dynamic user preferences
* Use C++ libraries and APIs to manage model structures and parameters
* Implement a C++ program to solve image classification tasks with LeNet architecture

If you feel this book is for you, get your [copy](https://www.amazon.com/dp/B0881XCLY8) today!

<a href="https://www.packtpub.com/?utm_source=github&utm_medium=banner&utm_campaign=GitHubBanner"><img src="https://raw.githubusercontent.com/PacktPublishing/GitHub/master/GitHub.png" alt="https://www.packtpub.com/" border="5" /></a>

## Instructions and Navigations
All of the code is organized into folders.

The code will look like the following:
```
class Network {
  public:
    Network(const std::string& snapshot_path,
            const std::string& synset_path,
            torch::DeviceType device_type);
    std::string Classify(const at::Tensor& image);
  private:
    torch::DeviceType device_type_;
    Classes classes_;
    torch::jit::script::Module model_;
};

```

**Following is what you need for this book:**
You will find this C++ machine learning book useful if you want to get started with machine learning algorithms and techniques using the popular C++ language. As well as being a useful first course in machine learning with C++, this book will also appeal to data analysts, data scientists, and machine learning developers who are looking to implement different machine learning models in production using varied datasets and examples. Working knowledge of the C++ programming language is mandatory to get started with this book.

With the following software and hardware list you can run all code files present in the book (Chapter 1-13).

### Software and Hardware List

| Chapter  | Software required                                                                    | OS required                        |
| -------- | -------------------------------------------------------------------------------------| -----------------------------------|
| 1 - 13   |   C++, Python 3.5+, Anroid SDK, Google Cloud Platform (trial version)                | Windows, Mac OS X, and Linux (Any) |

We also provide a PDF file that has color images of the screenshots/diagrams used in this book. [Click here to download it](https://static.packt-cdn.com/downloads/9781789955330_ColorImages.pdf).


### Related products <Other books you may enjoy>
* The C++ Workshop [[Packt]](https://www.packtpub.com/programming/the-c-workshop?utm_source=github&utm_medium=repository&utm_campaign=9781839216626) [[Amazon]](https://www.amazon.com/Workshop-New-Interactive-Approach-Learning-ebook/dp/B082451SZ9)

* Expert C++ [[Packt]](https://www.packtpub.com/programming/mastering-c-programming?utm_source=github&utm_medium=repository&utm_campaign=9781838552657) [[Amazon]](https://www.amazon.com/Expert-proficient-programmer-learning-practices-ebook/dp/B085G6VVW2)

## Get to Know the Author
**Kirill Kolodiazhnyi**
is a seasoned software engineer with expertise in custom software development. He has several years of experience building machine learning models and data products using C++. He holds a bachelor degree in Computer Science from the Kharkiv National University of Radio-Electronics. He currently works in Kharkiv, Ukraine where he lives with his wife and daughter.

### Suggestions and Feedback
[Click here](https://docs.google.com/forms/d/e/1FAIpQLSdy7dATC6QmEL81FIUuymZ0Wy9vH1jHkvpY57OiMeKGqib_Ow/viewform) if you have any feedback or suggestions.

