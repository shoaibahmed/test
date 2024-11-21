
#include <span>
#include<vector>
#include<torch/extension.h>

torch::Tensor bow_computation(
    const torch::Tensor &input_ids,
    int vocab_size,
    int bos_token_id,
    const std::vector<int> &prediction_heads,
    const std::vector<torch::Tensor> &bag_weights,
    const std::string &multihead_token_weighting_scheme,
    const torch::Tensor &idf
) {
    int max_pred_horizon = *std::max_element(prediction_heads.begin(), prediction_heads.end());
    int target_len = input_ids.size(0) - max_pred_horizon;
    int num_heads = (int) prediction_heads.size();

    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor all_targets = torch::zeros({num_heads, target_len, vocab_size}, options);

    #pragma omp parallel for
    for (int head_index = 0; head_index < num_heads; head_index++) {
        int pred_horizon = prediction_heads[head_index];
        torch::Tensor weight_vector = bag_weights[head_index];  // Now uses the provided weight vector

        #pragma omp parallel for
        for (int i = 0; i < target_len; ++i) {
            int start_idx = i + 1;
            int end_idx = start_idx + pred_horizon;

            torch::Tensor current_input_chunk = input_ids.slice(0, start_idx, end_idx);
            torch::Tensor relative_freq = torch::zeros({vocab_size}, options);

            torch::Tensor bos_mask = current_input_chunk.eq(bos_token_id);
            current_input_chunk = current_input_chunk.where(
                bos_mask.cumsum(0) == 0,
                bos_token_id
            );

            relative_freq.scatter_add_(0, current_input_chunk, weight_vector);
            if (multihead_token_weighting_scheme == "idf") {
                relative_freq *= idf;
                relative_freq /= relative_freq.sum();
            }
            all_targets[head_index][i].copy_(relative_freq);
        }
    }

    return all_targets;
}

torch::Tensor incremental_bow_computation(
    const torch::Tensor &input_ids,
    int vocab_size,
    int bos_token_id,
    const std::vector<int> &prediction_heads,
    const std::vector<torch::Tensor> &bag_weights,
    const std::vector<double> &p_values,
    const std::string &multihead_token_weighting_scheme,
    const torch::Tensor &idf
) {
    int max_pred_horizon = *std::max_element(prediction_heads.begin(), prediction_heads.end());
    int target_len = input_ids.size(0) - max_pred_horizon;
    int num_heads = (int) prediction_heads.size();

    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor all_targets = torch::zeros({num_heads, target_len, vocab_size}, options);

    assert(input_ids.is_contiguous());
    const int64_t input_size = input_ids.size(0);
    int64_t* input_ids_data = input_ids.data_ptr<int64_t>();

    #pragma omp parallel for
    for (int head_index = 0; head_index < num_heads; head_index++) {
        int pred_horizon = prediction_heads[head_index];
        torch::Tensor weight_vector = bag_weights[head_index];

        double p;
        if (multihead_token_weighting_scheme == "truncated_exp") {
            p = p_values[head_index];
        }

        torch::Tensor relative_freq = torch::zeros({vocab_size}, options);
        float* relative_freq_data = relative_freq.data_ptr<float>();
        bool bos_masking = false;

        float w_1 = weight_vector[0].item<float>();
        int w_vector_size = weight_vector.size(0);
        float w_W = weight_vector[w_vector_size - 1].item<float>();

        for (int i = 0; i < target_len; ++i) {
            int start_idx = i + 1;
            int last_idx = i + pred_horizon;
            int end_idx = last_idx + 1;

            int64_t removed_token = input_ids_data[i];
            int64_t added_token = input_ids_data[last_idx];

            if ((i == 0) || (removed_token == bos_token_id)) {
                bos_masking = false;
                relative_freq.zero_();

                torch::Tensor current_input_chunk = input_ids.slice(0, start_idx, end_idx);
                torch::Tensor bos_mask = current_input_chunk.eq(bos_token_id);
                if (bos_mask.any().item<bool>()) {
                    current_input_chunk = current_input_chunk.where(
                        bos_mask.cumsum(0) == 0,
                        bos_token_id
                    );
                    bos_masking = true;
                }
                relative_freq.scatter_add_(0, current_input_chunk, weight_vector);
            } else {
                if (added_token == bos_token_id) {
                    bos_masking = true;
                }
                relative_freq_data[removed_token] -= w_1;
                if (multihead_token_weighting_scheme == "truncated_exp") {
                    relative_freq.mul_(1.0 / (1.0 - p));
                }
                if (bos_masking) {
                    relative_freq_data[bos_token_id] += w_W;
                } else {
                    relative_freq_data[added_token] += w_W;
                }
            }
            if (multihead_token_weighting_scheme == "idf") {
                torch::Tensor weighted_freq = relative_freq * idf;
                weighted_freq /= weighted_freq.sum();
                all_targets[head_index][i].copy_(weighted_freq);
            } else {
                all_targets[head_index][i].copy_(relative_freq);
            }
        }
    }

    return all_targets;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bow_computation", &bow_computation, "Non-Incremental BoW computation function");
    m.def("incremental_bow_computation", &incremental_bow_computation, "Incremental BoW computation function");
}
