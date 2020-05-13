#include <jni.h>
#include <string>
#include <iostream>

#include <torch/script.h>
#include <caffe2/serialize/read_adapter_interface.h>

#include <android/asset_manager_jni.h>
#include <android/asset_manager.h>

#include <android/log.h>

#define  LOGD(...)  __android_log_print(ANDROID_LOG_DEBUG, "CAMERA_TAG", __VA_ARGS__)

namespace {

    class ModelReader : public caffe2::serialize::ReadAdapterInterface {
    public:
        explicit ModelReader(const std::vector<char> &buf) : buf_(&buf) {}

        ~ModelReader() override {};

        virtual size_t size() const override {
            return buf_->size();
        }

        virtual size_t read(uint64_t pos, void *buf, size_t n, const char *what)
        const override {
            std::copy_n(buf_->begin() + pos, n, reinterpret_cast<char *>(buf));
            return n;
        }

    private:
        const std::vector<char> *buf_;
    };

    class ImageClassifier {
    public:
        using Classes = std::map<size_t, std::string>;

        ImageClassifier() = default;

        void InitSynset(std::istream &stream) {
            LOGD("Init synset start OK");
            classes_.clear();
            if (stream) {
                std::string line;
                std::string id;
                std::string label;
                std::string token;
                size_t idx = 1;
                while (std::getline(stream, line)) {
                    auto pos = line.find_first_of(" ");
                    id = line.substr(0, pos);
                    label = line.substr(pos + 1);
                    classes_.insert({idx, label});
                    ++idx;
                }
            }
            LOGD("Init synset finish OK");
        }

        void InitModel(const std::vector<char> &buf) {
            model_ = torch::jit::load(std::make_unique<ModelReader>(buf), at::kCPU);
        }

        std::string Classify(const at::Tensor &image) {
            std::vector<torch::jit::IValue> inputs;
            inputs.emplace_back(image);
            at::Tensor output = model_.forward(inputs).toTensor();

            LOGD("Output size %d %d %d", static_cast<int>(output.ndimension()),
                 static_cast<int>(output.size(0)),
                 static_cast<int>(output.size(1)));

            auto max_result = output.squeeze().max(0);
            auto max_index = std::get<1>(max_result).item<int64_t>();
            auto max_value = std::get<0>(max_result).item<float>();

            max_index += 1;

            return std::to_string(max_index) + " - " + std::to_string(max_value) + " - " +
                   classes_[static_cast<size_t>(max_index)];
        }

    private:
        Classes classes_;
        torch::jit::script::Module model_;
    };

    template<typename CharT, typename TraitsT = std::char_traits<CharT> >
    struct VectorStreamBuf : public std::basic_streambuf<CharT, TraitsT> {
        explicit VectorStreamBuf(std::vector<CharT> &vec) {
            this->setg(vec.data(), vec.data(), vec.data() + vec.size());
        }
    };

    std::vector<char> ReadAsset(AAssetManager *asset_manager, const std::string &name) {
        std::vector<char> buf;
        AAsset *asset = AAssetManager_open(asset_manager, name.c_str(), AASSET_MODE_UNKNOWN);
        if (asset != nullptr) {
            LOGD("Open asset %s OK", name.c_str());
            off_t buf_size = AAsset_getLength(asset);
            buf.resize(buf_size + 1, 0);
            auto num_read = AAsset_read(asset, buf.data(), buf_size);
            LOGD("Read asset %s OK", name.c_str());

            if (num_read == 0)
                buf.clear();
            AAsset_close(asset);
            LOGD("Close asset %s OK", name.c_str());
        }
        return buf;
    }

    ImageClassifier g_image_classifier;
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_camera2_MainActivity_initClassifier(
        JNIEnv *env, jobject /*self*/, jobject j_asset_manager) {
    AAssetManager *asset_manager = AAssetManager_fromJava(env, j_asset_manager);
    if (asset_manager != nullptr) {
        LOGD("initClassifier start OK");


        auto model = ReadAsset(asset_manager, "model.pt");
        if (!model.empty()) {
            g_image_classifier.InitModel(model);
        }

        auto synset = ReadAsset(asset_manager, "synset.txt");
        if (!synset.empty()) {
            VectorStreamBuf<char> stream_buf(synset);
            std::istream is(&stream_buf);
            g_image_classifier.InitSynset(is);
        }
        LOGD("initClassifier finish OK");
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_camera2_MainActivity_destroyClassifier(
        JNIEnv *env, jobject /*self*/) {
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_camera2_MainActivity_classifyBitmap(
        JNIEnv *env, jobject /*self*/, jintArray pixels, jint width, jint height) {
    LOGD("classifyBitmap start OK");

    jboolean is_copy = 0;
    jint *pixels_buf = env->GetIntArrayElements(pixels, &is_copy);

    auto channel_size = static_cast<size_t>(width * height);
    using ChannelData = std::vector<float>;
    size_t channels_num = 3; // RGB image
    std::vector<ChannelData> image_data(channels_num);
    for (size_t i = 0; i < channels_num; ++i) {
        image_data[i].resize(channel_size);
    }

    LOGD("Alloc image, channel size %d OK", channel_size);

    // split original image
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            auto pos = x + y * width;
            auto pixel_color = static_cast<uint32_t>(pixels_buf[pos]); // ARGB format
            uint32_t mask{0x000000FF};

            for (size_t i = 0; i < channels_num; ++i) {
                uint32_t shift = i * 8;
                uint32_t channel_value = (pixel_color >> shift) & mask;
                image_data[channels_num - (i + 1)][pos] = static_cast<float>(channel_value);
            }
        }
    }

    env->ReleaseIntArrayElements(pixels, pixels_buf, 0);
    LOGD("Read image OK");


    // create image channel tensors
    std::vector<int64_t> channel_dims = {height, width};

    LOGD("Tensor required size %d %d", height, width);

    std::vector<at::Tensor> channel_tensor;
    at::TensorOptions options(at::kFloat);
    options = options.device(at::kCPU).requires_grad(false);

    for (size_t i = 0; i < channels_num; ++i) {
        channel_tensor.emplace_back(
                torch::from_blob(image_data[i].data(), at::IntArrayRef(channel_dims),
                                 options).clone());
    }

    LOGD("Tensor size %d %d", static_cast<int>(channel_tensor[0].size(0)),
         static_cast<int>(channel_tensor[0].size(1)));
    LOGD("Image to tensors OK");

    // normalize
    std::vector<float> mean{0.485f, 0.456f, 0.406f};
    std::vector<float> stddev{0.229f, 0.224f, 0.225f};


    for (size_t i = 0; i < channels_num; ++i) {
        channel_tensor[i] = ((channel_tensor[i] / 255.0f) - mean[i]) / stddev[i];
    }

    LOGD("Tensor size %d %d", static_cast<int>(channel_tensor[0].size(0)),
         static_cast<int>(channel_tensor[0].size(1)));
    LOGD("Normalize tensors OK");


    // stack channels
    auto image_tensor = at::stack(channel_tensor);
    image_tensor = image_tensor.unsqueeze(0);

    LOGD("Image concatenate OK %d %d %d %d", static_cast<int>(image_tensor.size(0)),
         static_cast<int>(image_tensor.size(1)),
         static_cast<int>(image_tensor.size(2)),
         static_cast<int>(image_tensor.size(3)));


    std::string hello = g_image_classifier.Classify(image_tensor);


    LOGD("classifyBitmap finish OK");
    return env->NewStringUTF(hello.c_str());
}