#include <cassert>
#include <iostream>
#include <llama.cpp>
#include <memory>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <vector>

namespace {
std::shared_ptr<spdlog::logger> logger() {
  static auto logger_ = spdlog::stdout_color_mt("ndk_echo");
  return logger_;
}
} // namespace

namespace {
struct llama_model_deleter {
  void operator()(llama_model *model) noexcept { llama_free_model(model); }
};
using unique_llama_model = std::unique_ptr<llama_model, llama_model_deleter>;

class llama_cpp_model {
public:
  static llama_cpp_model load_from_file(std::string const &model_file_path,
                                        size_t n_threads, size_t n_gpu_layers) {
    // model
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = n_gpu_layers;
    unique_llama_model model(
        llama_load_model_from_file(model_file_path.c_str(), model_params));
    if (!model) {
      throw std::runtime_error("wrong model_path: " + model_file_path);
    }

    return llama_cpp_model(std::move(model));
  }

  llama_model *get() { return model_.get(); }

  /*
std::vector<float> calc_next_token_logits(std::vector<int> const &input_ids) {
  auto *logits_data = llama_get_logits_ith(ctx_.get(), batch.n_tokens - 1);
  std::vector<float> logits(vocab_size_);
  std::copy(logits_data, logits_data + vocab_size_, logits.begin());
  return logits;
}
  */

private:
  llama_cpp_model(unique_llama_model &&model) : model_(std::move(model)) {
    vocab_size_ = llama_n_vocab(model_.get());
  }

  /*
  bool is_first(std::vector<int> const &input_ids) {
    static std::vector<int> input_ids_before_backup = std::vector<int>();
    std::vector<int> input_ids_before = input_ids_before_backup;
    if (input_ids_before_backup.empty()) {
      input_ids_before_backup = input_ids;
      return true;
    }
    input_ids_before_backup = input_ids;
    if (input_ids_before.size() > input_ids.size()) {
      return true;
    }
    for (size_t i = 0; i < input_ids_before.size(); ++i) {
      if (input_ids_before[i] != input_ids[i]) {
        return true;
      }
    }
    return false;
  }
  */

  unique_llama_model model_;

  size_t vocab_size_;
};
} // namespace

namespace lm {

using float32_t = float;
using float16_t = uint16_t;

enum class dtype_t {
  dtype_int64,
  dtype_float32,
  dtype_float16,
};

constexpr size_t k_bits_per_byte = 8;

class dtype_traits {
public:
  dtype_traits(std::string const &name, size_t size_in_bits)
      : name_(name), size_in_bits_(size_in_bits) {}

  std::string name() const { return name_; }
  size_t size_in_bits() const { return size_in_bits_; }

private:
  std::string name_;
  size_t size_in_bits_;
};

size_t size_in_bytes(dtype_traits const &traits) {
  return traits.size_in_bits() / k_bits_per_byte;
}

dtype_traits get_dtype_traits(dtype_t dtype) {
  if (dtype == dtype_t::dtype_int64) {
    return dtype_traits("i64", 64);
  } else if (dtype == dtype_t::dtype_float16) {
    return dtype_traits("f16", 16);
  } else if (dtype == dtype_t::dtype_float32) {
    return dtype_traits("f32", 32);
  }
  throw std::runtime_error("not implemented dtype");
}

constexpr size_t k_max_dims = 4;

template <typename T> T *pointer_cast(std::byte *data) {
  return static_cast<T *>(static_cast<void *>(data));
}
template <typename T> std::byte *byte_pointer_cast(T *data) {
  return static_cast<std::byte *>(static_cast<void *>(data));
}

class context {
public:
  std::byte *allocate(size_t size_in_bytes) {
    size_t size = size_in_bytes / sizeof(std::max_align_t);
    if (size_in_bytes % sizeof(std::max_align_t) != 0) {
      size += 1;
    }
    buffers_.emplace_back(std::make_unique<std::vector<std::max_align_t>>(
        std::vector<std::max_align_t>(size)));
    return byte_pointer_cast(buffers_.back()->data());
  }

private:
  std::vector<std::unique_ptr<std::vector<std::max_align_t>>> buffers_;
};

class tensor {
public:
  tensor(dtype_t dtype, std::array<int64_t, k_max_dims> const &n_elements,
         std::array<size_t, k_max_dims> const &n_strides_in_bytes,
         std::byte *data)
      : dtype_(dtype), n_elements_(n_elements),
        n_strides_in_bytes_(n_strides_in_bytes), data_(data) {}

  std::byte *data() const { return data_; }

  int64_t ne(size_t i) const { return n_elements_[i]; }
  int64_t nb(size_t i) const { return n_strides_in_bytes_[i]; }

  dtype_t dtype() const { return dtype_; }

private:
  dtype_t dtype_;
  std::array<int64_t, k_max_dims> n_elements_;
  std::array<size_t, k_max_dims> n_strides_in_bytes_;
  std::byte *data_;
};

template <typename T>
T &at(tensor const &t, int64_t ne0 = 0, int64_t ne1 = 0, int64_t ne2 = 0,
      int64_t ne3 = 0) {
  return *pointer_cast<T>(t.data() + ne0 * t.nb(0) + ne1 * t.nb(1) +
                          ne2 * t.nb(2) + ne3 * t.nb(3));
}

std::string to_string(tensor const &t) {
  return get_dtype_traits(t.dtype()).name() + " " + std::to_string(t.ne(0)) +
         " " + std::to_string(t.ne(1)) + " " + std::to_string(t.ne(2)) + " " +
         std::to_string(t.ne(3)) + " | " + std::to_string(t.nb(0)) + " " +
         std::to_string(t.nb(1)) + " " + std::to_string(t.nb(2)) + " " +
         std::to_string(t.nb(3));
}

tensor make_new_tensor_4d(context &ctx, dtype_t dtype, //
                          int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
  size_t dtype_size_in_bytes = size_in_bytes(get_dtype_traits(dtype));
  std::byte *allocated_buffer =
      ctx.allocate(dtype_size_in_bytes * ne0 * ne1 * ne2 * ne3);
  return tensor(dtype, {ne0, ne1, ne2, ne3},
                {dtype_size_in_bytes,             //
                 dtype_size_in_bytes * ne0,       //
                 dtype_size_in_bytes * ne0 * ne1, //
                 dtype_size_in_bytes * ne0 * ne1 * ne2},
                allocated_buffer);
}

tensor make_new_tensor_3d(context &ctx, dtype_t dtype, //
                          int64_t ne0, int64_t ne1, int64_t ne2) {
  return make_new_tensor_4d(ctx, dtype, ne0, ne1, ne2, 1);
}

tensor make_new_tensor_2d(context &ctx, dtype_t dtype, //
                          int64_t ne0, int64_t ne1) {
  return make_new_tensor_4d(ctx, dtype, ne0, ne1, 1, 1);
}

tensor make_new_tensor_1d(context &ctx, dtype_t dtype, //
                          int64_t ne0) {
  return make_new_tensor_4d(ctx, dtype, ne0, 1, 1, 1);
}

tensor embedding(context &ctx, tensor const &token_embed_weight,
                 tensor const &input_ids) {
  // embed_weight: embed_dim x vocab_size
  // input_ids: seq_len x batch_size
  // out: embed_dim x seq_len x batch_size
  size_t embed_dim = token_embed_weight.ne(0);
  size_t batch_size = input_ids.ne(1);
  size_t seq_len = input_ids.ne(0);
  tensor out = make_new_tensor_3d(ctx, token_embed_weight.dtype(), embed_dim,
                                  seq_len, batch_size);
  std::cout << to_string(out) << std::endl;
  for (size_t bi = 0; bi < batch_size; ++bi) {
    for (size_t pos = 0; pos < seq_len; ++pos) {
      int64_t input_id = *pointer_cast<int64_t>(
          input_ids.data() + bi * input_ids.nb(1) + pos * input_ids.nb(0));
      /*
      std::cout << input_id << " copy " << input_id * token_embed_weight.nb(1)
                << " " << (input_id + 1) * token_embed_weight.nb(1) << " "
                << pos * out.nb(1) + bi * out.nb(2) << std::endl;
      */
      std::copy(token_embed_weight.data() + input_id * token_embed_weight.nb(1),
                token_embed_weight.data() +
                    (input_id + 1) * token_embed_weight.nb(1),
                out.data() + pos * out.nb(1) + bi * out.nb(2));
    }
  }
  return out;
}

// tensor rope() {}

tensor rms_norm(context &ctx, tensor const &x, float32_t eps) {
  tensor out = make_new_tensor_3d(ctx, x.dtype(), x.ne(0), x.ne(1), x.ne(2));
  for (size_t bi = 0; bi < x.ne(2); ++bi) {
    for (size_t pos = 0; pos < x.ne(1); ++pos) {
      float32_t sum = 0;
      for (size_t i = 0; i < x.ne(0); ++i) {
        sum += at<float32_t>(x, i, pos, bi) * at<float32_t>(x, i, pos, bi);
      }
      float32_t var = sum / out.ne(0);
      std::cout << "var " << var << std::endl;
      for (size_t i = 0; i < x.ne(0); ++i) {
        at<float32_t>(out, i, pos, bi) =
            at<float32_t>(x, i, pos, bi) / std::sqrt(var + eps);
      }
    }
  }
  return out;
}

tensor mul(context &ctx, tensor const &w, tensor const &x) {
  tensor out = make_new_tensor_4d(ctx, x.dtype(), x.ne(0), x.ne(1), x.ne(2), x.ne(3));
  for (size_t e3 = 0; e3 < x.ne(3); ++e3) {
    for (size_t e2 = 0; e2 < x.ne(2); ++e2) {
      for (size_t e1 = 0; e1 < x.ne(1); ++e1) {
        for (size_t e0 = 0; e0 < x.ne(0); ++e0) {
          at<float32_t>(out, e0, e1, e2, e3) =
              at<float32_t>(w, e0 % w.ne(0), e1 % w.ne(1), e2 % w.ne(2),
                            e3 % w.ne(3)) *
              at<float32_t>(x, e0, e1, e2, e3);
        }
      }
    }
  }
  return out;
}

tensor add(context &ctx, tensor const &a, tensor const &b) {
  tensor out = make_new_tensor_4d(ctx, a.dtype(), a.ne(0), a.ne(1), a.ne(2), a.ne(3));
  for (size_t e3 = 0; e3 < a.ne(3); ++e3) {
    for (size_t e2 = 0; e2 < a.ne(2); ++e2) {
      for (size_t e1 = 0; e1 < a.ne(1); ++e1) {
        for (size_t e0 = 0; e0 < a.ne(0); ++e0) {
          at<float32_t>(out, e0, e1, e2, e3) =
              at<float32_t>(a, e0, e1, e2, e3) +
              at<float32_t>(b, e0 % b.ne(0), e1 % b.ne(1), e2 % b.ne(2),
                            e3 % b.ne(3));
        }
      }
    }
  }
  return out;
}

tensor mul_mat(context &ctx, tensor const &a, tensor const &b) {
  tensor out = make_new_tensor_3d(ctx, a.dtype(), a.ne(1), b.ne(1), b.ne(2), b.ne(3));
  for (size_t e3 = 0; e3 < a.ne(3); ++e3) {
    for (size_t e2 = 0; e2 < a.ne(2); ++e2) {

      for (size_t e1 = 0; e1 < a.ne(1); ++e1) {
        for (size_t e0 = 0; e0 < a.ne(0); ++e0) {
          at<float32_t>(out, i, pos, bi) =
              at<float32_t>(w, i % w.ne(0), pos % w.ne(1), bi % w.ne(2)) *
              at<float32_t>(x, i, pos, bi);
        }
      }

    }
  }
  return out;
}

tensor convert_float32_ggml_tensor_to_lm_tensor(context &ctx,
                                                ggml_tensor const &gt) {
  tensor out = make_new_tensor_4d(ctx, dtype_t::dtype_float32, gt.ne[0],
                                  gt.ne[1], gt.ne[2], gt.ne[3]);
  assert(gt.type == 0);
  std::cout << gt.ne[0] << " " << gt.ne[1] << " " << gt.ne[2] << " " << gt.ne[3]
            << std::endl;
  for (size_t e3 = 0; e3 < gt.ne[3]; ++e3) {
    for (size_t e2 = 0; e2 < gt.ne[2]; ++e2) {
      for (size_t e1 = 0; e1 < gt.ne[1]; ++e1) {
        for (size_t e0 = 0; e0 < gt.ne[0]; ++e0) {
          lm::at<float32_t>(out, e0, e1, e2, e3) = *lm::pointer_cast<float32_t>(
              static_cast<std::byte *>(gt.data) + //
              e0 * gt.nb[0] + e1 * gt.nb[1] + e2 * gt.nb[2] + e3 * gt.nb[3]);
        }
      }
    }
  }
  return out;
}

} // namespace lm

int main() {
  std::cout << "hello" << std::endl;

  lm::context ctx;

  size_t batch_size = 1;
  size_t seq_len = 4;
  size_t vocab_size = 16;
  size_t embed_dim = 8;
  lm::tensor token_embed_weight = lm::make_new_tensor_2d(
      ctx, lm::dtype_t::dtype_float32, embed_dim, vocab_size);
  std::cout << lm::to_string(token_embed_weight) << std::endl;
  lm::tensor input_ids = lm::make_new_tensor_2d(ctx, lm::dtype_t::dtype_int64,
                                                seq_len, batch_size);
  at<int64_t>(input_ids, 0) = 0;
  at<int64_t>(input_ids, 1) = 1;
  at<int64_t>(input_ids, 2) = 2;
  at<int64_t>(input_ids, 3) = 3;
  std::cout << lm::to_string(input_ids) << std::endl;
  std::cout << "(0, 0) " << at<int64_t>(input_ids, 0, 0) << std::endl;
  std::cout << "(1, 0) " << at<int64_t>(input_ids, 1, 0) << std::endl;
  std::cout << "(2, 0) " << at<int64_t>(input_ids, 2, 0) << std::endl;
  std::cout << "(3, 0) " << at<int64_t>(input_ids, 3, 0) << std::endl;
  /*
  lm::tensor embd_inp = embedding(ctx, token_embed_weight, input_ids);
  std::cout << lm::to_string(embd_inp) << std::endl;
  */

  std::string model_file_path =
      "/home/okada/storage/sandman_llm_train/pipeline_out/hoge-wf/"
      "convert_to_gguf/model.gguf";
  //"quantized_model.gguf";
  int n_gpu_layers = 0;
  auto model =
      llama_cpp_model::load_from_file(model_file_path, 1, n_gpu_layers);
  std::cout << model.get()->tok_embd->name << std::endl;
  std::cout << model.get()->tok_embd->type << std::endl;
  lm::tensor tok_embd =
      lm::convert_float32_ggml_tensor_to_lm_tensor(ctx, *model.get()->tok_embd);
  std::cout << lm::at<lm::float32_t>(tok_embd, 0) << std::endl;
  std::cout << lm::at<lm::float32_t>(tok_embd, 1) << std::endl;

  std::cout << "\n";

  std::cout << lm::to_string(tok_embd) << std::endl;
  lm::tensor embd_inp2 = embedding(ctx, tok_embd, input_ids);
  std::cout << lm::to_string(embd_inp2) << std::endl;
  std::cout << lm::at<lm::float32_t>(embd_inp2, 0) << std::endl;
  std::cout << lm::at<lm::float32_t>(embd_inp2, 1) << std::endl;
  std::cout << lm::at<lm::float32_t>(embd_inp2, 2) << std::endl;
  std::cout << lm::at<lm::float32_t>(embd_inp2, 3) << std::endl;
  std::cout << lm::at<lm::float32_t>(embd_inp2, 4) << std::endl;
  std::cout << lm::at<lm::float32_t>(embd_inp2, 5) << std::endl;

  lm::float32_t eps = 0.01;
  lm::tensor norm_inp = rms_norm(ctx, embd_inp2, eps);
  std::cout << lm::to_string(norm_inp) << std::endl;
  std::cout << lm::at<lm::float32_t>(norm_inp, 0) << std::endl;
  std::cout << lm::at<lm::float32_t>(norm_inp, 1) << std::endl;
  std::cout << lm::at<lm::float32_t>(norm_inp, 2) << std::endl;
  std::cout << lm::at<lm::float32_t>(norm_inp, 3) << std::endl;
  /*

  // llama_backend_init();
  // llama_numa_init(GGML_NUMA_STRATEGY_DISABLED);

  std::string model_file_path =
      "/home/okada/android_llama_cpp/cordova-plugin-ndk-echo/data/"
      "quantized_model.gguf";
  int n_gpu_layers = 99;
  auto model =
      llama_cpp_model::load_from_file(model_file_path, 1, n_gpu_layers);
  std::cout << "layers.size() " << model.get()->layers.size() << std::endl;
  std::cout << model.get()->output_norm->name << std::endl;
  std::cout << model.get()->tok_embd->name << std::endl;
  std::cout << model.get()->tok_embd->ne[0] << std::endl;
  std::cout << model.get()->tok_embd->ne[1] << std::endl;
  std::cout << model.get()->tok_embd->ne[2] << std::endl;
  std::cout << model.get()->tok_embd->ne[3] << std::endl;
  std::cout << model.get()->tok_embd->nb[0] << std::endl;
  std::cout << model.get()->tok_embd->nb[1] << std::endl;
  std::cout << model.get()->tok_embd->nb[2] << std::endl;
  std::cout << model.get()->tok_embd->nb[3] << std::endl;

  //embedding(model.get()->tok_embd, input_ids);
  size_t vocab_size = model.get()->tok_embd->ne[1];
  lm::tensor input_ids(lm::dtype_t::dtype_int32, {vocab_size, 1, 1, 1},
  {sizeof(int32_t), 1, 1, 1}, nullptr);
  */
}
