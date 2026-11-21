#include "models.h"
#include "../llama-memory-hybrid-iswa.h"

void llama_model_inkling2::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_ATTENTION_QK_CLIP,           hparams.f_qk_clip);
    ml.get_key(LLM_KV_ATTN_RES_BLOCK_SIZE,         hparams.attn_res_block_size);
    ml.get_key(LLM_KV_SHORTCONV_L_CACHE,           hparams.n_shortconv_l_cache);

    // relative-position bias, and the log-scaled positions of the full-attention layers
    ml.get_key(LLM_KV_ATTENTION_RELATIVE_LENGTH, hparams.n_relative);
    ml.get_key(LLM_KV_ATTENTION_RELATIVE_EXTENT, hparams.n_relative_extent);
    ml.get_key(LLM_KV_ATTENTION_RELATIVE_CLIP,   hparams.f_relative_clip);
    ml.get_key(LLM_KV_ATTENTION_LOG_SCALE_FLOOR, hparams.n_log_scale_floor);
    ml.get_key(LLM_KV_ATTENTION_LOG_SCALE_ALPHA, hparams.f_log_scale_alpha);

    // the embedding and output tensors are padded past the vocabulary
    ml.get_key(LLM_KV_VOCAB_SIZE,          hparams.n_vocab_padded);
    ml.get_key(LLM_KV_VOCAB_SIZE_UNPADDED, hparams.n_vocab_unpadded);

    // MoE
    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all);

    ml.get_key(LLM_KV_ACTIVATION_SITU_BETA,        hparams.situ_beta);
    ml.get_key(LLM_KV_ACTIVATION_SITU_LINEAR_BETA, hparams.situ_linear_beta);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale, false);

    hparams.f_final_logit_softcapping = 0.0f;
    ml.get_key(LLM_KV_FINAL_LOGIT_SOFTCAPPING, hparams.f_final_logit_softcapping, false);

    // sliding window
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,         hparams.n_swa);
    ml.get_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.is_swa_impl);

    uint32_t n_pattern = 0;
    ml.get_arr_n(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, n_pattern);

    if (n_pattern != hparams.n_layer() || !hparams.n_relative || !hparams.n_relative_extent || !hparams.n_swa ||
        !hparams.attn_res_block_size || !hparams.n_log_scale_floor || hparams.n_shortconv_l_cache < 2 ||
        hparams.n_shortconv_l_cache > 64 || hparams.n_relative_extent > INT32_MAX || hparams.n_swa > INT32_MAX ||
        !hparams.n_vocab_unpadded || hparams.n_vocab_unpadded > hparams.n_vocab_padded) {
        throw std::runtime_error("inkling2: invalid dimensions, vocabulary, or layer pattern");
    }

    for (float value : {hparams.f_norm_rms_eps, hparams.f_qk_clip, hparams.f_relative_clip, hparams.situ_beta, hparams.situ_linear_beta}) {
        if (!std::isfinite(value) || value <= 0.0f) {
            throw std::runtime_error("inkling2: normalization, clipping and scaling parameters must be finite and positive");
        }
    }

    if (!std::isfinite(hparams.f_log_scale_alpha) || hparams.f_log_scale_alpha < 0.0f) {
        throw std::runtime_error("inkling2: invalid logarithmic scale");
    }

    hparams.swa_type                = LLAMA_SWA_TYPE_STANDARD;
    hparams.expert_router_f32       = true;
    hparams.expert_weights_norm_eps = 1e-20f;

    // no RoPE, the relative-position bias carries the positions
    hparams.rope_pattern.fill(false);

    // the convolution state keeps K, V, the attention output and the FFN output of the widest layer
    uint64_t n_channels_max = 0;

    for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
        if (!hparams.n_head_kv(il) || hparams.n_head(il) % hparams.n_head_kv(il) ||
            hparams.n_embd_head_k(il) != hparams.n_embd_head_v(il) || !hparams.n_expert_used(il) ||
            hparams.n_expert_used(il) > hparams.n_expert || !hparams.n_ff_exp(il)) {
            throw std::runtime_error("inkling2: invalid attention or expert dimensions");
        }

        n_channels_max = std::max<uint64_t>(n_channels_max, 2ull * hparams.n_embd_k_gqa(il) + 2ull * hparams.n_embd);
    }

    if (n_channels_max * (hparams.n_shortconv_l_cache - 1) > UINT32_MAX) {
        throw std::runtime_error("inkling2: convolution state is too large");
    }

    hparams.n_sconv_channels = n_channels_max;

    type = LLM_TYPE_UNKNOWN;
}

void llama_model_inkling2::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    if (hparams.n_vocab_unpadded != vocab.n_tokens()) {
        throw std::runtime_error("inkling2: tokenizer size must equal the unpadded vocabulary size");
    }

    const int64_t n_vocab_padded = hparams.n_vocab_padded;
    const int64_t n_rel          = hparams.n_relative;
    const int64_t n_conv         = hparams.n_shortconv_l_cache;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,      "weight"),    {n_embd, n_vocab_padded}, 0);
    tok_norm = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD_NORM, "weight", 0), {n_embd}, 0);

    // output
    output           = create_tensor(tn(LLM_TENSOR_OUTPUT,           "weight"), {n_embd, n_vocab_padded}, 0);
    output_norm      = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM,      "weight"), {n_embd}, 0);
    output_res_query = create_tensor(tn(LLM_TENSOR_OUTPUT_RES_QUERY, "weight"), {n_embd}, 0);

    for (int i = 0; i < n_layer_phys(); ++i) {
        auto & layer = layers[i];

        const int64_t n_head      = hparams.n_head(i);
        const int64_t n_head_kv   = hparams.n_head_kv(i);
        const int64_t n_embd_head = hparams.n_embd_head_k(i);
        const int64_t n_ff_exp    = hparams.n_ff_exp(i);
        const int64_t n_extent    = hparams.is_swa(i) ? hparams.n_swa : hparams.n_relative_extent;

        // attention
        layer.wq        = create_tensor(tn(LLM_TENSOR_ATTN_Q,    "weight", i), {n_embd, n_embd_head*n_head}, 0);
        layer.wk        = create_tensor(tn(LLM_TENSOR_ATTN_K,    "weight", i), {n_embd, n_embd_head*n_head_kv}, 0);
        layer.wv        = create_tensor(tn(LLM_TENSOR_ATTN_V,    "weight", i), {n_embd, n_embd_head*n_head_kv}, 0);
        layer.wo        = create_tensor(tn(LLM_TENSOR_ATTN_OUT,  "weight", i), {n_embd_head*n_head, n_embd}, 0);
        layer.wqkv_gate = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "weight", i), {n_embd, n_embd_head*n_head}, 0);

        layer.attn_q_clip    = create_tensor(tn(LLM_TENSOR_ATTN_Q_CLIP,    "weight", i), {n_embd_head, n_head}, 0);
        layer.attn_sink_proj = create_tensor(tn(LLM_TENSOR_ATTN_SINK_PROJ, "weight", i), {n_embd, n_head}, 0);

        // relative-position bias
        layer.attn_rel      = create_tensor(tn(LLM_TENSOR_ATTN_REL,      "weight", i), {n_embd, n_rel*n_head}, 0);
        layer.attn_rel_bias = create_tensor(tn(LLM_TENSOR_ATTN_REL_BIAS, "weight", i), {n_rel, n_extent}, 0);
        layer.attn_rel_clip = create_tensor(tn(LLM_TENSOR_ATTN_REL_CLIP, "weight", i), {n_rel, n_head}, 0);

        // short convolutions
        layer.attn_k_conv    = create_tensor(tn(LLM_TENSOR_ATTN_K_CONV,    "weight", i), {n_conv, n_embd_head*n_head_kv}, 0);
        layer.attn_v_conv    = create_tensor(tn(LLM_TENSOR_ATTN_V_CONV,    "weight", i), {n_conv, n_embd_head*n_head_kv}, 0);
        layer.attn_post_conv = create_tensor(tn(LLM_TENSOR_ATTN_POST_CONV, "weight", i), {n_conv, n_embd}, 0);
        layer.ffn_post_conv  = create_tensor(tn(LLM_TENSOR_FFN_POST_CONV,  "weight", i), {n_conv, n_embd}, 0);

        if (hparams.is_swa(i)) {
            layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
        } else {
            // full-attention layers: separate Q, K and V input norms, and input-dependent taps for the K and V convolutions
            layer.attn_q_inp_norm  = create_tensor(tn(LLM_TENSOR_ATTN_Q_INP_NORM,  "weight", i), {n_embd}, 0);
            layer.attn_k_inp_norm  = create_tensor(tn(LLM_TENSOR_ATTN_K_INP_NORM,  "weight", i), {n_embd}, 0);
            layer.attn_v_inp_norm  = create_tensor(tn(LLM_TENSOR_ATTN_V_INP_NORM,  "weight", i), {n_embd}, 0);
            layer.attn_k_conv_proj = create_tensor(tn(LLM_TENSOR_ATTN_K_CONV_PROJ, "weight", i), {n_embd, n_conv*n_head_kv}, 0);
            layer.attn_v_conv_proj = create_tensor(tn(LLM_TENSOR_ATTN_V_CONV_PROJ, "weight", i), {n_embd, n_conv*n_head_kv}, 0);
        }

        // block attention residuals
        if (i > 0) {
            layer.attn_res_query = create_tensor(tn(LLM_TENSOR_ATTN_RES_QUERY, "weight", i), {n_embd, hparams.is_swa(i) ? 1 : 3}, 0);
        }
        layer.ffn_res_query = create_tensor(tn(LLM_TENSOR_FFN_RES_QUERY, "weight", i), {n_embd}, 0);

        // MoE
        layer.ffn_norm         = create_tensor(tn(LLM_TENSOR_FFN_NORM,         "weight", i), {n_embd}, 0);
        layer.ffn_gate_inp     = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,     "weight", i), {n_embd, n_expert}, 0);
        layer.ffn_exp_probs_b  = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B,  "bias",   i), {n_expert}, TENSOR_NOT_REQUIRED);
        layer.ffn_gate_up_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_UP_EXPS, "weight", i), {n_embd, 2*n_ff_exp, n_expert}, 0);
        layer.ffn_down_exps    = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS,    "weight", i), {n_ff_exp, n_embd, n_expert}, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_inkling2::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(params);
}

// scale each row down to an L2 norm of at most threshold, after the optional per-channel scale, then multiply by the optional tau
ggml_tensor * llama_model_inkling2::graph::clip(ggml_tensor * cur, ggml_tensor * scale, ggml_tensor * tau, float threshold, int il) {
    if (cparams.fused_inkling2) {
        cur = ggml_inkling2_clip(ctx0, cur, scale, tau, threshold, 1e-6f);
        res->add_fused_node({LLM_FUSED_OP_INKLING2, cur, il});
        return cur;
    }

    if (scale) {
        cur = ggml_mul(ctx0, cur, scale);
    }

    ggml_tensor * norm = ggml_sqrt(ctx0, ggml_sum_rows(ctx0, ggml_sqr(ctx0, cur)));

    // x*min(1, t/(norm + eps)) = x/max(1, (norm + eps)/t)
    norm = ggml_clamp(ctx0, ggml_scale_bias(ctx0, norm, 1.0f/threshold, 1e-6f/threshold), 1.0f, INFINITY);

    cur = ggml_div(ctx0, cur, norm);

    return tau ? ggml_mul(ctx0, cur, tau) : cur;
}

// attention over the depth: mix the normalized blocks [n_embd, n, n_tokens] (or null) and rms_norm(cur) with learned queries
// cur_norm is rms_norm(cur) when the caller has it, or null; the fused node of layer il is registered when il >= 0
ggml_tensor * llama_model_inkling2::graph::mix(ggml_tensor * blocks, ggml_tensor * cur, ggml_tensor * cur_norm, ggml_tensor * query, int il) {
    if (cparams.fused_inkling2) {
        ggml_tensor * out = ggml_inkling2_mix(ctx0, blocks, cur, query, 1.0f/sqrtf(n_embd), 1e-6f);
        if (il >= 0) {
            res->add_fused_node({LLM_FUSED_OP_INKLING2, out, il});
        }
        return out;
    }

    return mix_states(stack(blocks, cur_norm ? cur_norm : ggml_rms_norm(ctx0, cur, 1e-6f)), query);
}

// attention over the depth: mix the normalized states [n_embd, n, n_tokens] with learned queries
ggml_tensor * llama_model_inkling2::graph::mix_states(ggml_tensor * states, ggml_tensor * query) {
    // one token needs no repeat, the query broadcasts
    ggml_tensor * queries = n_tokens == 1 ? query :
        ggml_repeat(ctx0, query, ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_embd, query->ne[1], n_tokens));

    ggml_tensor * scores = ggml_mul_mat(ctx0, states, queries);
    scores = ggml_soft_max_ext(ctx0, scores, nullptr, 1.0f/sqrtf(n_embd), 0.0f);

    return ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, states)), scores);
}

// the states of a mix: the normalized blocks, then the normalized cur
ggml_tensor * llama_model_inkling2::graph::stack(ggml_tensor * blocks, ggml_tensor * cur) {
    cur = ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tokens);
    return blocks ? ggml_concat(ctx0, blocks, cur, 1) : cur;
}

// causal short convolution with a residual; dynamic adds input-dependent taps per head
// the last n_conv - 1 inputs of each sequence live in the recurrent state at channel offset `offset`
ggml_tensor * llama_model_inkling2::graph::convolve(
        ggml_tensor * cur,
        ggml_tensor * weight,
        ggml_tensor * dynamic,
        ggml_tensor * state,
        ggml_tensor * history,
        llm_graph_input_rs * inp,
        int64_t offset,
        int il,
        ggml_tensor * residual) {
    const int64_t n_conv       = hparams.n_shortconv_l_cache;
    const int64_t n_channels   = cur->ne[0];
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;
    const int64_t n_seqs       = ubatch.n_seqs;
    const float   scale        = 1.0f/sqrtf((dynamic ? 2 : 1)*n_conv + 1);

    ggml_tensor * past = ggml_view_3d(ctx0, history, n_conv - 1, n_channels, n_seqs, history->nb[1], history->nb[2], offset*history->nb[1]);

    // the fused op also returns the last n_conv - 1 inputs, the new state
    if (cparams.fused_inkling2 && cparams.n_rs_seq == 0 && weight->type == GGML_TYPE_F32) {
        ggml_tensor * res_conv = ggml_inkling2_conv(ctx0, cur, past, weight, dynamic, residual, scale);
        res->add_fused_node({LLM_FUSED_OP_INKLING2, res_conv, il});

        ggml_tensor * x_new = ggml_view_3d(ctx0, res_conv, n_conv - 1, n_channels, n_seqs,
                (n_conv - 1)*sizeof(float), (n_conv - 1)*n_channels*sizeof(float), n_channels*n_tokens*sizeof(float));
        build_conv_state_store(inp, state, x_new, n_conv - 1, offset*(n_conv - 1)*ggml_element_size(state));

        return ggml_view_2d(ctx0, res_conv, n_channels, n_tokens, n_channels*sizeof(float), 0);
    }

    ggml_tensor * x = ggml_transpose(ctx0, ggml_reshape_3d(ctx0, cur, n_channels, n_seq_tokens, n_seqs));
    x = ggml_concat(ctx0, past, x, 0);

    build_conv_state_store(inp, state, x, n_conv - 1, offset*(n_conv - 1)*ggml_element_size(state));

    ggml_tensor * out = ggml_ssm_conv(ctx0, x, weight);
    out = ggml_reshape_2d(ctx0, out, n_channels, n_tokens);

    if (dynamic) {
        const int64_t n_heads = dynamic->ne[0]/n_conv;

        dynamic = ggml_reshape_4d(ctx0, dynamic, n_conv, n_heads, n_seq_tokens, n_seqs);

        // the inputs by channel, [n_channels, n_conv - 1 + n_seq_tokens, n_seqs]
        ggml_tensor * xt = ggml_cont(ctx0, ggml_transpose(ctx0, x));

        // all terms first, so the adds below are adjacent in the graph and a backend can fuse them
        std::vector<ggml_tensor *> terms;
        for (int64_t lag = 0; lag < n_conv; ++lag) {
            ggml_tensor * tap    = ggml_view_4d(ctx0, dynamic, 1, n_heads, n_seq_tokens, n_seqs, dynamic->nb[1], dynamic->nb[2], dynamic->nb[3], lag*dynamic->nb[0]);
            ggml_tensor * window = ggml_view_4d(ctx0, xt, n_channels/n_heads, n_heads, n_seq_tokens, n_seqs,
                    (n_channels/n_heads)*xt->nb[0], xt->nb[1], xt->nb[2], (n_conv - 1 - lag)*xt->nb[1]);

            terms.push_back(ggml_reshape_2d(ctx0, ggml_mul(ctx0, window, tap), n_channels, n_tokens));
            ggml_build_forward_expand(gf, terms.back());
        }

        for (ggml_tensor * term : terms) {
            out = ggml_add(ctx0, out, term);
        }
    }

    out = ggml_scale(ctx0, ggml_add(ctx0, out, cur), scale);

    return residual ? ggml_add(ctx0, residual, out) : out;
}

// the fused controller op reads the plain weights and normalizes its inputs itself
// it is a mat-vec per token, so large batches keep the mul_mat graph
bool llama_model_inkling2::graph::fused_control(const loop_layer & loop) const {
    if (!cparams.fused_inkling2 || n_tokens > std::max<int64_t>(8, ubatch.n_seqs)) {
        return false;
    }

    for (int ctrl = 0; ctrl < LOOP_CTRL_COUNT; ++ctrl) {
        for (ggml_tensor * w : {loop.ctrl_down[ctrl], loop.ctrl_gain[ctrl], loop.ctrl_forget[ctrl]}) {
            if (w->type != GGML_TYPE_F32 && w->type != GGML_TYPE_F16 && w->type != GGML_TYPE_BF16) {
                return false;
            }
            if (hdmd && hdmd->rot.count(w)) {
                return false;
            }
            for (const auto & lora : *loras) {
                if (lora.first->get_weight(w)) {
                    return false;
                }
            }
        }
    }

    return true;
}

// mix a cross-pass message into a looped layer input: (1 - forget)*cur + gain*message, both gated on cur and message
// cur_norm and message_norm are cur and message after an RMS norm, or null for the fused op of layer il
ggml_tensor * llama_model_inkling2::graph::control(ggml_tensor * cur, ggml_tensor * cur_norm, ggml_tensor * message, ggml_tensor * message_norm,
        const loop_layer & loop, int ctrl, int il) {
    if (!cur_norm) {
        ggml_tensor * out = ggml_inkling2_ctrl(ctx0, cur, message, loop.ctrl_down[ctrl], loop.ctrl_gain[ctrl], loop.ctrl_forget[ctrl], 1e-6f);
        res->add_fused_node({LLM_FUSED_OP_INKLING2, out, il});
        return out;
    }

    ggml_tensor * gate = ggml_concat(ctx0, cur_norm, message_norm, 0);
    gate = build_lora_mm(loop.ctrl_down[ctrl], gate);

    ggml_tensor * gain   = ggml_tanh(ctx0, build_lora_mm(loop.ctrl_gain[ctrl],   gate));
    ggml_tensor * forget = ggml_tanh(ctx0, build_lora_mm(loop.ctrl_forget[ctrl], gate));

    // the gates go second, so that no new view comes between a tanh and its mul and a backend can fuse them
    return ggml_add(ctx0, ggml_sub(ctx0, cur, ggml_mul(ctx0, cur, forget)), ggml_mul(ctx0, message, gain));
}

ggml_tensor * llama_model_inkling2::graph::build_inputs() {
    inp  = build_inp_mem_hybrid_iswa_rel(hparams.n_relative_extent, hparams.n_swa);
    attn = inp->get_attn();
    recr = inp->get_recr();

    ggml_tensor * inp_pos = build_inp_pos();
    inp_out_ids = build_inp_out_ids();

    // log-scaled positions: 1 + alpha*log(max(1, (pos + 1)/floor)), multiplies q and r of the full-attention layers
    tau = ggml_scale_bias(ctx0, ggml_cast(ctx0, inp_pos, GGML_TYPE_F32), 1.0f, 1.0f);
    tau = ggml_log(ctx0, ggml_clamp(ctx0, ggml_scale(ctx0, tau, 1.0f/hparams.n_log_scale_floor), 1.0f, INFINITY));
    tau = ggml_reshape_3d(ctx0, ggml_scale_bias(ctx0, tau, hparams.f_log_scale_alpha, 1.0f), 1, 1, n_tokens);

    ggml_tensor * cur = build_inp_embd(model.tok_embd);

    // important: do not normalize raw embeddings input (i.e. encoded image embeddings)
    if (ubatch.token) {
        cur = build_norm(cur, model.tok_norm, nullptr, LLM_NORM_RMS, -1);
    }

    return cur;
}

ggml_tensor * llama_model_inkling2::graph::build_layer(int il, int ip, ggml_tensor * cur, ggml_tensor *& blocks,
        ggml_tensor * messages, const loop_layer * loop) {
    GGML_ASSERT(!messages == !loop);

    const auto & layer = model.layers[il];

    const bool    is_swa       = hparams.is_swa(il);
    const int64_t n_head       = hparams.n_head(il);
    const int64_t n_head_kv    = hparams.n_head_kv(il);
    const int64_t n_embd_head  = hparams.n_embd_head_k(il);
    const int64_t n_embd_k_gqa = n_embd_head*n_head_kv;

    // row j of a [n_embd, n, n_tokens] tensor, [n_embd, n_tokens]
    auto row = [&](ggml_tensor * t, int64_t j) {
        return ggml_view_2d(ctx0, t, n_embd, n_tokens, t->nb[2], j*t->nb[1]);
    };

    ggml_tensor * state   = recr->mctx->get_r_l(il);
    ggml_tensor * history = build_rs(recr, state, hparams.n_embd_r(), ubatch.n_seqs);
    history = ggml_reshape_3d(ctx0, history, hparams.n_shortconv_l_cache - 1, hparams.n_sconv_channels, ubatch.n_seqs);

    ggml_tensor * inpSA    = cur;
    ggml_tensor * inp_norm = cparams.fused_inkling2 ? nullptr : ggml_rms_norm(ctx0, cur, 1e-6f);

    // attention inputs: SWA layers mix one, full-attention layers one each for Q, K and V
    ggml_tensor * mixed = layer.attn_res_query ? mix(blocks, cur, inp_norm, layer.attn_res_query, ip) : ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tokens);
    ggml_tensor * inputs[3];
    for (int j = 0; j < 3; ++j) {
        inputs[j] = row(mixed, std::min<int64_t>(j, mixed->ne[1] - 1));
    }

    // the fused controllers need no normalized inputs
    const bool fused = messages && fused_control(*loop);

    ggml_tensor * messages_norm = messages && !fused ? ggml_rms_norm(ctx0, messages, 1e-6f) : nullptr;
    if (messages) {
        ggml_tensor * mixed_norm = fused ? nullptr : ggml_rms_norm(ctx0, mixed, 1e-6f);

        // the messages make the Q, K and V inputs of a SWA layer differ
        for (int j = 0; j < 3; ++j) {
            const int     ctrl = LOOP_CTRL_Q + j;
            const int64_t k    = std::min<int64_t>(j, mixed->ne[1] - 1);
            inputs[j] = control(inputs[j], fused ? nullptr : row(mixed_norm, k), row(messages, ctrl), fused ? nullptr : row(messages_norm, ctrl), *loop, ctrl, ip);
        }
    }

    if (is_swa && messages) {
        for (int j = 0; j < 3; ++j) {
            inputs[j] = build_norm(inputs[j], layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        }
    } else if (is_swa) {
        inputs[0] = build_norm(inputs[0], layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        inputs[1] = inputs[2] = inputs[0];
    } else {
        inputs[0] = build_norm(inputs[0], layer.attn_q_inp_norm, nullptr, LLM_NORM_RMS, il);
        inputs[1] = build_norm(inputs[1], layer.attn_k_inp_norm, nullptr, LLM_NORM_RMS, il);
        inputs[2] = build_norm(inputs[2], layer.attn_v_inp_norm, nullptr, LLM_NORM_RMS, il);
    }

    // self-attention
    {
        ggml_tensor * Qcur = build_lora_mm(layer.wq, inputs[0]);
        ggml_tensor * Kcur = convolve(build_lora_mm(layer.wk, inputs[1]), layer.attn_k_conv,
                is_swa ? nullptr : build_lora_mm(layer.attn_k_conv_proj, inputs[1]), state, history, recr, 0, ip);
        ggml_tensor * Vcur = convolve(build_lora_mm(layer.wv, inputs[2]), layer.attn_v_conv,
                is_swa ? nullptr : build_lora_mm(layer.attn_v_conv_proj, inputs[2]), state, history, recr, n_embd_k_gqa, ip);
        ggml_tensor * Rcur = build_lora_mm(layer.attn_rel, inputs[0]);

        // full-attention layers scale q and r by the log-scaled positions
        ggml_tensor * tau_l = is_swa ? nullptr : tau;

        Qcur = clip(ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens), layer.attn_q_clip, tau_l, hparams.f_qk_clip, ip);
        Kcur = clip(ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), nullptr,           nullptr, hparams.f_qk_clip, ip);
        Vcur = ggml_rms_norm(ctx0, ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens), hparams.f_norm_rms_eps);
        Rcur = clip(ggml_reshape_3d(ctx0, Rcur, hparams.n_relative, n_head, n_tokens), layer.attn_rel_clip, tau_l, hparams.f_relative_clip, ip);

        cb(Qcur, "Qcur", il);
        cb(Kcur, "Kcur", il);
        cb(Vcur, "Vcur", il);
        cb(Rcur, "Rcur", il);

        ggml_tensor * rel_pos = is_swa ? inp->self_rel_pos_swa : inp->self_rel_pos;

        const int64_t n_stream = (is_swa ? attn->self_attn_mask_swa.raw : attn->self_attn_mask.raw)->ne[3];
        const int64_t n_kv     = rel_pos->ne[0];

        ggml_tensor * sinks = build_lora_mm(layer.attn_sink_proj, inputs[0]);

        // the sink logit t is per query: exp(s - t)/(1 + sum(exp(s - t))) = exp(s)/(exp(t) + sum(exp(s))), with a zero sink
        ggml_tensor * kq_b;
        if (cparams.fused_inkling2) {
            const auto & attn_mask = is_swa ? attn->self_attn_mask_swa : attn->self_attn_mask;
            GGML_ASSERT(attn_mask.rows == nullptr);

            kq_b = ggml_inkling2_rel_bias(ctx0, layer.attn_rel_bias, Rcur, sinks, rel_pos, attn_mask.raw, n_stream);
            res->add_fused_node({LLM_FUSED_OP_INKLING2, kq_b, ip});
        } else {
            // project r on the bias table first, then gather by distance: only the gathered bias is n_kv x n_tokens x n_head
            ggml_tensor * table = ggml_mul_mat(ctx0, ggml_pad(ctx0, layer.attn_rel_bias, 0, 1, 0, 0), Rcur);

            table = ggml_sub(ctx0, table, ggml_reshape_3d(ctx0, sinks, 1, n_head, n_tokens));
            table = ggml_cont(ctx0, ggml_permute(ctx0, table, 1, 0, 2, 3));

            kq_b = ggml_get_rows(ctx0, table, rel_pos);
            kq_b = ggml_reshape_4d(ctx0, kq_b, n_head, n_kv, n_tokens/n_stream, n_stream);
            kq_b = ggml_cont(ctx0, ggml_permute(ctx0, kq_b, 2, 0, 1, 3));
        }
        cb(kq_b, "kq_b", il);

        if (!sinks_zero || sinks_zero->ne[0] != n_head) {
            sinks_zero = ggml_fill(ctx0, ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, n_head), 0.0f);
        }

        cur = build_attn(inp,
                nullptr, nullptr, nullptr,
                Qcur, Kcur, Vcur, kq_b, sinks_zero, 1.0f/sqrtf(n_embd_head), il, cparams.fused_inkling2);

        cur = ggml_mul(ctx0, cur, ggml_sigmoid(ctx0, build_lora_mm(layer.wqkv_gate, inputs[0])));
        // the first layer of each block starts a new residual stream, the earlier one joins the mixed blocks
        const bool new_block = ip % hparams.attn_res_block_size == 0;

        cur = convolve(build_lora_mm(layer.wo, cur), layer.attn_post_conv, nullptr, state, history, recr, 2*n_embd_k_gqa, ip, new_block ? nullptr : inpSA);
        cb(cur, "attn_out", il);
    }

    if (ip % hparams.attn_res_block_size == 0) {
        inp_norm = ggml_reshape_3d(ctx0, inp_norm ? inp_norm : ggml_rms_norm(ctx0, inpSA, 1e-6f), n_embd, 1, n_tokens);
        blocks   = blocks ? ggml_concat(ctx0, inp_norm, blocks, 1) : inp_norm;
    }

    ggml_tensor * ffn_inp = cur;
    cb(ffn_inp, "ffn_inp", il);

    // MoE
    cur = ggml_reshape_2d(ctx0, mix(blocks, cur, nullptr, layer.ffn_res_query, ip), n_embd, n_tokens);
    if (messages) {
        cur = control(cur, fused ? nullptr : ggml_rms_norm(ctx0, cur, 1e-6f), row(messages, LOOP_CTRL_FFN),
                fused ? nullptr : row(messages_norm, LOOP_CTRL_FFN), *loop, LOOP_CTRL_FFN, ip);
    }
    cur = build_norm(cur, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);

    cur = build_moe_ffn(cur,
            layer.ffn_gate_inp,
            nullptr,
            nullptr,
            layer.ffn_down_exps,
            layer.ffn_exp_probs_b,
            hparams.n_expert,
            hparams.n_expert_used(il),
            LLM_FFN_SITU, true,
            hparams.expert_weights_scale,
            LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID,
            il,
            nullptr,
            layer.ffn_gate_up_exps,
            layer.ffn_up_exps_s,
            layer.ffn_gate_exps_s,
            layer.ffn_down_exps_s);

    cur = convolve(cur, layer.ffn_post_conv, nullptr, state, history, recr, 2*n_embd_k_gqa + n_embd, ip, ffn_inp);
    cb(cur, "l_out", il);

    return cur;
}

void llama_model_inkling2::graph::build_output(ggml_tensor * cur, ggml_tensor * blocks) {
    cur = ggml_reshape_2d(ctx0, mix(blocks, cur, nullptr, model.output_res_query, -1), n_embd, n_tokens);
    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);

    if (inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    cur = build_lora_mm(model.output, cur);
    ggml_prec_set_acc(cur, GGML_PREC_F32);

    if (hparams.f_final_logit_softcapping != 0.0f) {
        const float cap = hparams.f_final_logit_softcapping;

        cur = ggml_scale(ctx0, ggml_tanh(ctx0, ggml_scale(ctx0, cur, 1.0f/cap)), cap);
    }

    // drop the padded vocabulary
    cur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, hparams.n_vocab_unpadded, cur->ne[1], cur->nb[1], 0));

    cb(cur, "result_output", -1);
    res->t_logits = cur;
}

llama_model_inkling2::graph::graph(const llm_graph_params & params) : llm_graph_context(params) {
    ggml_tensor * cur = build_inputs();

    ggml_tensor * blocks = nullptr;
    for (int il = 0; il < n_layer; ++il) {
        cur = build_layer(il, il, cur, blocks);
    }

    build_output(cur, blocks);
}
