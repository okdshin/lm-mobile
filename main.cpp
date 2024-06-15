#include <llama_mobile.hpp>
#include <load_tensor_from_safetensors.hpp>
#include <map>

int main() {
  lm::context ctx;
  std::map<std::string, lm::tensor> params_raw = lm::load_safetensors_from_file(
      ctx, "/home/okada/Downloads/model-00001-of-00004.safetensors");
  std::map<std::string, lm::tensor> params;
  for (auto const &[name, param] : params_raw) {
    std::cout << name << std::endl;
    // std::cout << lm::to_string(lm::cast_bfloat16_to_float32(ctx, param)) <<
    // std::endl;
    if (param.dtype() == lm::dtype_t::dtype_float32) {
      params.emplace(name, param);
    } else if (param.dtype() == lm::dtype_t::dtype_bfloat16) {
      params.emplace(name, lm::cast_bfloat16_to_float32(ctx, param));
    } else {
      throw std::runtime_error("unsupported dtype: " +
                               lm::get_dtype_traits(param.dtype()).name());
    }
  }

  lm::float32_t eps = 1e-5;
  int64_t num_attention_heads = 32;
  int64_t num_key_value_heads = 8;
  int64_t head_dim = 128;
  int64_t hidden_size = 4096;
  lm::float32_t rope_theta = 500000;

  int64_t batch_size = 1;
  int64_t seq_len = 4;
  int64_t vocab_size = 16;
  int64_t embed_dim = 8;
  lm::tensor input_ids =
      lm::make_new_tensor(ctx, lm::dtype_t::dtype_int64, seq_len, batch_size);
  lm::at<int64_t>(input_ids, 0) = 0;
  lm::at<int64_t>(input_ids, 1) = 1;
  lm::at<int64_t>(input_ids, 2) = 2;
  lm::at<int64_t>(input_ids, 3) = 3;

  lm::tensor token_embed_weight = params["model.embed_tokens.weight"];
  std::cout << "token_embed_weight " << lm::to_string(token_embed_weight)
            << std::endl;

  lm::tensor hidden_state = lm::embedding(ctx, token_embed_weight, input_ids);
  std::cout << "hidden_state " << lm::to_string(hidden_state) << std::endl;

  for (size_t i = 0; i < 1; ++i) {
    lm::tensor residual = lm::copy_contiguous(ctx, hidden_state);
    lm::tensor input_layer_norm_weight =
        params["model.layers.0.input_layernorm.weight"];
    hidden_state = rms_norm(ctx, hidden_state, eps);
    hidden_state = mul(ctx, hidden_state, input_layer_norm_weight);
    std::cout << "hidden_state " << lm::to_string(hidden_state) << std::endl;

    // self att
    lm::tensor wq = params["model.layers.0.self_attn.q_proj.weight"];
    std::cout << "wq " << lm::to_string(wq) << std::endl;
    lm::tensor q = mul_mat(ctx, wq, hidden_state);
    std::cout << "q " << lm::to_string(q) << std::endl;

    lm::tensor wk = params["model.layers.0.self_attn.k_proj.weight"];
    std::cout << "wk " << lm::to_string(wk) << std::endl;
    lm::tensor k = mul_mat(ctx, wk, hidden_state);
    std::cout << "k " << lm::to_string(k) << std::endl;

    lm::tensor wv = params["model.layers.0.self_attn.v_proj.weight"];
    std::cout << "wv " << lm::to_string(wv) << std::endl;
    lm::tensor v = mul_mat(ctx, wv, hidden_state);
    std::cout << "v " << lm::to_string(v) << std::endl;

    q = lm::copy_contiguous(
        ctx, q.reshape({head_dim, num_attention_heads, seq_len, batch_size})
                 .transpose({0, 2, 1, 3}));
    k = lm::copy_contiguous(
        ctx, k.reshape({head_dim, num_key_value_heads, seq_len, batch_size})
                 .transpose({0, 2, 1, 3}));
    /*
    v = lm::copy_contiguous(
        ctx, v.reshape({head_dim, num_key_value_heads, seq_len, batch_size})
                 .transpose({2, 0, 1, 3}));
    */
    v = lm::copy_contiguous(
        ctx, v.reshape({head_dim, num_key_value_heads, seq_len, batch_size})
                 .transpose({2, 0, 1, 3}));
    std::cout << "q_rt " << lm::to_string(q) << std::endl;
    std::cout << "k_rt " << lm::to_string(k) << std::endl;
    std::cout << "v_rt " << lm::to_string(v) << std::endl;

    auto check = [](lm::tensor const& t, std::string const& gt_file_path) {
        std::ifstream ifs(gt_file_path);
        std::vector<float> gt;
        std::copy(std::istream_iterator<float>(ifs), std::istream_iterator<float>(), std::back_inserter(gt));
        std::cout << "gt.size " << gt.size() << std::endl;
        for(size_t i = 0; i < gt.size(); ++i) {
            float val = *(lm::pointer_cast<float>(t.data()) + i);
            if(std::pow(val - gt[i], 2) > 1e-6) {
                std::cout << i << " " << val << " != " << gt[i] << std::endl;
            }
            else {
                std::cout << i << " ok" << std::endl;
            }
        }
    };

    lm::tensor position_ids =
        lm::make_new_tensor(ctx, lm::dtype_t::dtype_float32, seq_len, 1);
    for (size_t i = 0; i < seq_len; ++i) {
      lm::at<lm::float32_t>(position_ids, i) = i;
    }
    lm::float32_t base = 500000;
    auto [query_embed, key_embed] =
        lm::rope(ctx, position_ids, base, head_dim, q, k);
    //query_embed = lm::copy_contiguous(ctx, query_embed);
    //query_embed = lm::copy_contiguous(ctx, query_embed.reshape({128, 4, 4, 8}).transpose({0, 1, 3, 2})).reshape({128, 4, 32});
    std::cout << "query_embed " << lm::to_string(query_embed) << std::endl;
    //check(query_embed, "/home/okada/android_llama_cpp/llama-mobile/check/query_embed.txt");
    std::cout << "key_embed " << lm::to_string(key_embed) << std::endl;
    //check(key_embed, "/home/okada/android_llama_cpp/llama-mobile/check/key_embed.txt");

    lm::tensor attn_weights =
        lm::div(ctx, lm::mul_mat(ctx, key_embed, query_embed),
                lm::make_new_scalar(ctx, lm::dtype_t::dtype_float32,
                                    std::sqrt(head_dim)));
    std::cout << "(kq) attn_weights " << lm::to_string(attn_weights)
              << std::endl;

    // attn_weights = lm::apply_causal_mask(ctx, attn_weights.transpose({1, 0,
    // 2, 3}));
    attn_weights = lm::apply_causal_mask(ctx, attn_weights);
    attn_weights = softmax(ctx, attn_weights, 1.0e-8);
    std::cout << "(softmax qk) softmax attn_weights "
              << lm::to_string(attn_weights) << std::endl;
    //check(attn_weights, "/home/okada/android_llama_cpp/llama-mobile/check/attn_weight.txt");

    hidden_state = mul_mat(ctx, v, attn_weights);
    //check(hidden_state, "/home/okada/android_llama_cpp/llama-mobile/check/attn_weight_x_v.txt");
    /*
    hidden_state =
        lm::copy_contiguous(ctx, hidden_state.transpose({1, 2, 0, 3}));
    hidden_state.reshape({hidden_size, seq_len, batch_size});
    */
    hidden_state =
        lm::copy_contiguous(ctx, hidden_state.transpose({0, 2, 1, 3}));
    /*
    hidden_state =
        lm::copy_contiguous(ctx, hidden_state.transpose({2, 0, 1, 3}));
    */
    hidden_state.reshape({hidden_size, seq_len, batch_size});
    std::cout << "(attn_weights x v) hidden_state "
              << lm::to_string(hidden_state) << std::endl;

    for (int64_t e0 = 0; e0 < hidden_state.ne(0); ++e0) {
      std::cout << int(100 * at<lm::float32_t>(hidden_state, e0, 0, 0, 0))
                << ", ";
    }
    std::cout << std::endl;

    lm::tensor wo = params["model.layers.0.self_attn.o_proj.weight"];
    std::cout << "wo " << lm::to_string(wo) << std::endl;
    // hidden_state = mul_mat(ctx, wo.transpose({1, 0, 2, 3}), hidden_state);
    hidden_state = mul_mat(ctx, wo, hidden_state);
    std::cout << "o " << lm::to_string(hidden_state) << std::endl;
    //check(hidden_state, "/home/okada/android_llama_cpp/llama-mobile/check/attn_output.txt");

    hidden_state = lm::add(ctx, hidden_state, residual);
    std::cout << "hidden_state " << lm::to_string(hidden_state) << std::endl;

    lm::tensor residual2 = lm::copy_contiguous(ctx, hidden_state);
    lm::tensor post_attn_layer_norm_weight =
        params["model.layers.0.post_attention_layernorm.weight"];
    hidden_state = rms_norm(ctx, hidden_state, eps);
    hidden_state = mul(ctx, hidden_state, post_attn_layer_norm_weight);
    std::cout << "hidden_state " << lm::to_string(hidden_state) << std::endl;
    // mlp
    lm::tensor ffn_gate = params["model.layers.0.mlp.gate_proj.weight"];
    lm::tensor ffn_down = params["model.layers.0.mlp.down_proj.weight"];
    lm::tensor ffn_up = params["model.layers.0.mlp.up_proj.weight"];
    lm::tensor gate = mul_mat(ctx, ffn_gate, hidden_state);
    std::cout << "gate hidden_state " << lm::to_string(gate) << std::endl;
    // gate = act(ctx, gate);
    lm::tensor up = mul_mat(ctx, ffn_up, hidden_state);
    std::cout << "up hidden_state " << lm::to_string(up) << std::endl;
    hidden_state = mul(ctx, up, gate);
    lm::tensor down = mul_mat(ctx, ffn_down, hidden_state);
    hidden_state = lm::add(ctx, hidden_state, residual2);
    std::cout << "last hidden_state " << lm::to_string(hidden_state)
              << std::endl;
    check(hidden_state, "/home/okada/android_llama_cpp/llama-mobile/check/down_proj.txt");
    std::cout << "===" << std::endl;
    return 0;
  }

  /*
  lm::tensor output_norm_weight = lm::tensor ffn_gate =
  params["model.layers.0.mlp.gate_proj.weight"]; hidden_state = rms_norm(ctx,
  hidden_state, eps); hidden_state = mul(ctx, hidden_state, output_norm_weight);
  std::cout << "rms_norm " << lm::to_string(hidden_state) << std::endl;
  */
}
