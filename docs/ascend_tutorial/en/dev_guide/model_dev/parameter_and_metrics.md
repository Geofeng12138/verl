# Training Configuration Parameters and Metrics Description

Last updated: 07/02/2026.

For NPU-related features, see the [NPU Advanced Features Guide](https://github.com/verl-project/verl/blob/main/docs/ascend_tutorial/zh/feature_support/npu_advance_features.md).

verl manages all parameters through a hierarchical YAML configuration file, and all related configuration files are located in the `verl/trainer/config` directory.

---

## 1. Configuration Parameters

### 1.1 Common Configuration Parameters

The following parameters exist in both the FSDP and Megatron schemes and have the same meaning.

#### 1.1.1 Actor Optimizer Configuration

| Parameter | Default Value | Description |
|--------|--------|------|
| `actor_rollout_ref.actor.optim.lr` | `1.0e-06` | Actor learning rate |
| `actor_rollout_ref.actor.optim.lr_warmup_steps_ratio` | `0.0` | Ratio of learning rate warmup steps to total training steps |
| `actor_rollout_ref.actor.optim.total_training_steps` | `-1` | Total training steps; -1 indicates automatic calculation |
| `actor_rollout_ref.actor.optim.weight_decay` | `0.01` | Weight decay, used to prevent model overfitting |
| `actor_rollout_ref.actor.optim.lr_warmup_steps` | `-1` | Learning rate warmup steps; -1 indicates automatic calculation from the ratio |
| `actor_rollout_ref.actor.optim.betas` | `[0.9, 0.999]` | First and second order momentum coefficients of the Adam optimizer |
| `actor_rollout_ref.actor.optim.clip_grad` | `1.0` | Gradient clipping threshold |
| `actor_rollout_ref.actor.optim.override_optimizer_config` | `null` / `{}` | Overrides the optimizer configuration (null for FSDP, {} for Megatron) |

#### 1.1.2 Actor Policy Configuration

| Parameter | Default Value | Description |
|--------|--------|------|
| `actor_rollout_ref.actor.strategy` | `fsdp` / `megatron` | Training strategy. Use `fsdp` for the FSDP approach and `megatron` for the Megatron approach. |
| `actor_rollout_ref.actor.ppo_mini_batch_size` | `256` | Mini batch size for PPO training. |
| `actor_rollout_ref.actor.ppo_micro_batch_size` | `null` | Micro batch size for PPO training. |
| `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu` | `null` | PPO micro batch size per GPU. |
| `actor_rollout_ref.actor.use_dynamic_bsz` | `false` | Whether to use a dynamic batch size. |
| `actor_rollout_ref.actor.ppo_max_token_len_per_gpu` | `16384` | Maximum token length for PPO per GPU. |
| `actor_rollout_ref.actor.clip_ratio` | `0.2` | PPO clipping ratio that controls the policy update magnitude. The typical range is [0.1, 0.3]. |
| `actor_rollout_ref.actor.clip_ratio_low` | `0.2` | Lower bound clipping ratio for PPO. |
| `actor_rollout_ref.actor.clip_ratio_high` | `0.2` | Upper bound clipping ratio for PPO. |
| `actor_rollout_ref.actor.tau_pos` | `1.0` | Tau parameter for clipping positive advantages. |
| `actor_rollout_ref.actor.tau_neg` | `1.05` | Tau parameter for clipping negative advantages. |
| `actor_rollout_ref.actor.freeze_vision_tower` | `false` | Whether to freeze the vision tower (for multimodal models). |
| `actor_rollout_ref.actor.clip_ratio_c` | `3.0` | Upper bound constant for the clipping ratio. |
| `actor_rollout_ref.actor.loss_agg_mode` | `token-mean` | Loss aggregation mode. Options include `token-mean` and so on. |
| `actor_rollout_ref.actor.loss_scale_factor` | `null` | Loss scaling factor. |
| `actor_rollout_ref.actor.entropy_coeff` | `0` | Entropy regularization coefficient that controls the level of policy exploration. |
| `actor_rollout_ref.actor.calculate_entropy` | `false` | Whether to calculate policy entropy. |
| `actor_rollout_ref.actor.use_kl_loss` | `false` | Whether to use the KL divergence loss. |
| `actor_rollout_ref.actor.use_prefix_grouper` | `false` | Whether to use the prefix grouper. |
| `actor_rollout_ref.actor.use_torch_compile` | `true` | Whether to use `torch.compile` for acceleration. |
| `actor_rollout_ref.actor.kl_loss_coef` | `0.001` | KL loss coefficient. |
| `actor_rollout_ref.actor.kl_loss_type` | `low_var_kl` | KL loss type. Options include `low_var_kl` and so on. |
| `actor_rollout_ref.actor.ppo_epochs` | `1` | Number of PPO update epochs. |
| `actor_rollout_ref.actor.shuffle` | `false` | Whether to shuffle mini batches during training. |
| `actor_rollout_ref.actor.data_loader_seed` | `42` | Random seed for the data loader. |
| `actor_rollout_ref.actor.grad_clip` | `1.0` | Gradient clipping value. |
| `actor_rollout_ref.actor.ulysses_sequence_parallel_size` | `1` | Ulysses sequence parallel size. |
| `actor_rollout_ref.actor.entropy_from_logits_with_chunking` | `false` | Whether to calculate entropy from logits using chunking. |
| `actor_rollout_ref.actor.entropy_from_logits_chunk_size` | `2048` | Chunk size for entropy calculation. |
| `actor_rollout_ref.actor.entropy_checkpointing` | `false` | Whether to use gradient checkpointing for entropy calculation. |
| `actor_rollout_ref.actor.use_remove_padding` | Referenced from `model.use_remove_padding` | Whether to remove padding. |
| `actor_rollout_ref.actor.calculate_sum_pi_squared` | `false` | Whether to calculate the sum of squared policy probabilities. |
| `actor_rollout_ref.actor.sum_pi_squared_checkpointing` | `false` | Whether to use gradient checkpointing for the sum of squared policy probabilities calculation. |
| `actor_rollout_ref.actor.use_fused_kernels` | Referenced from `model.use_fused_kernels` | Whether to use fused kernels. |

#### 1.1.3 Policy Loss Configuration

| Parameter Name | Default Value | Description |
|--------|--------|------|
| `actor_rollout_ref.actor.policy_loss.loss_mode` | `vanilla` | Policy loss mode. Options include vanilla, clip_cov, kl_cov, dppo_tv, dppo_kl, gspo, sapo, geo_mean, cispo, gpg, bypass_mode, reinforce_is, and so on. |
| `actor_rollout_ref.actor.policy_loss.clip_cov_ratio` | `0.0002` | Covariance ratio for clip_cov mode. |
| `actor_rollout_ref.actor.policy_loss.clip_cov_lb` | `1.0` | Lower bound of covariance for clip_cov mode. |
| `actor_rollout_ref.actor.policy_loss.clip_cov_ub` | `5.0` | Upper bound of covariance for clip_cov mode. |
| `actor_rollout_ref.actor.policy_loss.kl_cov_ratio` | `0.0002` | Covariance ratio for kl_cov mode. |
| `actor_rollout_ref.actor.policy_loss.ppo_kl_coef` | `0.1` | PPO KL divergence coefficient. |

#### 1.1.4 Rollout Configuration

| Parameter | Default Value | Description |
|--------|--------|------|
| `actor_rollout_ref.rollout.name` | `???` | Rollout engine name, which you must specify |
| `actor_rollout_ref.rollout.mode` | `async` | Rollout mode. Options include `async`, `sync`, and so on |
| `actor_rollout_ref.rollout.nnodes` | `0` | Number of nodes used for rollout |
| `actor_rollout_ref.rollout.n_gpus_per_node` | Referenced from `trainer.n_gpus_per_node` | Number of GPUs per node |
| `actor_rollout_ref.rollout.temperature` | `1.0` | Sampling temperature, which controls generation randomness |
| `actor_rollout_ref.rollout.top_k` | `-1` | Top-K sampling parameter. `-1` means it is not enabled |
| `actor_rollout_ref.rollout.top_p` | `1` | Top-P (nucleus) sampling parameter |
| `actor_rollout_ref.rollout.prompt_length` | Referenced from `data.max_prompt_length` | Maximum prompt length |
| `actor_rollout_ref.rollout.response_length` | Referenced from `data.max_response_length` | Maximum response length |
| `actor_rollout_ref.rollout.dtype` | `bfloat16` | Data type for rollout inference |
| `actor_rollout_ref.rollout.gpu_memory_utilization` | `0.5` | GPU memory utilization, which is the proportion of GPU memory used during inference |
| `actor_rollout_ref.rollout.ignore_eos` | `false` | Whether to ignore the EOS token |
| `actor_rollout_ref.rollout.enforce_eager` | `false` | Whether to force the use of PyTorch eager mode |
| `actor_rollout_ref.rollout.cudagraph_capture_sizes` | `null` | List of CUDA Graph capture sizes |
| `actor_rollout_ref.rollout.free_cache_engine` | `true` | Whether to release the cache engine after each inference |
| `actor_rollout_ref.rollout.tensor_model_parallel_size` | `2` | Tensor parallel size for inference |
| `actor_rollout_ref.rollout.data_parallel_size` | `1` | Data parallel size for inference |
| `actor_rollout_ref.rollout.expert_parallel_size` | `1` | Expert parallel size for inference |
| `actor_rollout_ref.rollout.pipeline_model_parallel_size` | `1` | Pipeline parallel size for inference |
| `actor_rollout_ref.rollout.max_num_batched_tokens` | `8192` | Maximum number of batched tokens per step |
| `actor_rollout_ref.rollout.max_model_len` | `null` | Maximum model sequence length. `null` means it is inferred automatically |
| `actor_rollout_ref.rollout.max_num_seqs` | `1024` | Maximum number of concurrent samples for inference |
| `actor_rollout_ref.rollout.enable_chunked_prefill` | `true` | Whether to enable chunked prefill |
| `actor_rollout_ref.rollout.enable_prefix_caching` | `true` | Whether to enable prefix caching (KV cache reuse) |
| `actor_rollout_ref.rollout.logprobs_mode` | `processed_logprobs` | Logprobs computation mode |
| `actor_rollout_ref.rollout.scheduling_policy` | `fcfs` | Scheduling policy. Options include `fcfs`, and so on |
| `actor_rollout_ref.rollout.load_format` | `dummy` | Model loading format |
| `actor_rollout_ref.rollout.log_prob_micro_batch_size` | `null` | Micro batch size for log prob computation |
| `actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu` | `null` | Log prob micro batch size per GPU |
| `actor_rollout_ref.rollout.log_prob_use_dynamic_bsz` | Referenced from `actor.use_dynamic_bsz` | Whether to use a dynamic batch size for log prob |
| `actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu` | Referenced from `actor.ppo_max_token_len_per_gpu` | Maximum token length for log prob per GPU |
| `actor_rollout_ref.rollout.disable_log_stats` | `true` | Whether to disable inference log statistics |
| `actor_rollout_ref.rollout.do_sample` | `true` | Whether to perform sampling (`false` means greedy decoding) |
| `actor_rollout_ref.rollout.n` | `1` | Number of responses generated per prompt |
| `actor_rollout_ref.rollout.over_sample_rate` | `0` | Oversampling rate |
| `actor_rollout_ref.rollout.multi_stage_wake_up` | `false` | Whether to enable multi-stage wake-up |
| `actor_rollout_ref.rollout.calculate_log_probs` | `false` | Whether to calculate log probs during the rollout phase |
| `actor_rollout_ref.rollout.skip_tokenizer_init` | `true` | Whether to skip tokenizer initialization |
| `actor_rollout_ref.rollout.enable_rollout_routing_replay` | `false` | Whether to enable rollout routing replay |
| `actor_rollout_ref.rollout.quantization` | `null` | Quantization method |
| `actor_rollout_ref.rollout.quantization_config_file` | `null` | Path to the quantization configuration file |
| `actor_rollout_ref.rollout.layered_summon` | `false` | Whether to enable layered summon (FSDP scheme only) |

#### 1.1.5 Rollout Validation Sampling Configuration

| Parameter Name | Default Value | Description |
|--------|--------|------|
| `actor_rollout_ref.rollout.val_kwargs.top_k` | `-1` | Top-K sampling parameter for validation |
| `actor_rollout_ref.rollout.val_kwargs.top_p` | `1.0` | Top-P sampling parameter for validation |
| `actor_rollout_ref.rollout.val_kwargs.temperature` | `0` | Sampling temperature for validation; 0 indicates greedy decoding |
| `actor_rollout_ref.rollout.val_kwargs.n` | `1` | Number of responses generated per prompt during validation |
| `actor_rollout_ref.rollout.val_kwargs.do_sample` | `false` | Whether to sample during validation |

#### 1.1.6 Multi-Turn Configuration

| Parameter Name | Default Value | Description |
|--------|--------|------|
| `actor_rollout_ref.rollout.multi_turn.enable` | `false` | Whether to enable multi-turn conversation |
| `actor_rollout_ref.rollout.multi_turn.max_assistant_turns` | `null` | Maximum number of assistant turns |
| `actor_rollout_ref.rollout.multi_turn.tool_config_path` | `null` | Path to the tool configuration file |
| `actor_rollout_ref.rollout.multi_turn.max_user_turns` | `null` | Maximum number of user turns |
| `actor_rollout_ref.rollout.multi_turn.max_parallel_calls` | `1` | Maximum number of parallel tool calls |
| `actor_rollout_ref.rollout.multi_turn.max_tool_response_length` | `256` | Maximum length of tool responses |
| `actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side` | `middle` | Direction for truncating tool responses |
| `actor_rollout_ref.rollout.multi_turn.interaction_config_path` | `null` | Path to the interaction configuration file |
| `actor_rollout_ref.rollout.multi_turn.use_inference_chat_template` | `false` | Whether to use the inference chat template |
| `actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode` | `strict` | Tokenization sanity check mode |
| `actor_rollout_ref.rollout.multi_turn.format` | `hermes` | Multi-turn conversation format |
| `actor_rollout_ref.rollout.multi_turn.num_repeat_rollouts` | `null` | Number of repeated rollouts |

#### 1.1.7 Agent Configuration

| Parameter | Default Value | Description |
|--------|--------|------|
| `actor_rollout_ref.rollout.agent.num_workers` | `8` | Number of Agent worker processes |
| `actor_rollout_ref.rollout.agent.default_agent_loop` | `single_turn_agent` | Default Agent loop type |
| `actor_rollout_ref.rollout.agent.agent_loop_config_path` | `null` | Path to the Agent loop configuration file |
| `actor_rollout_ref.rollout.agent.custom_async_server.path` | `null` | Path to the custom asynchronous server |
| `actor_rollout_ref.rollout.agent.custom_async_server.name` | `null` | Name of the custom asynchronous server |

#### 1.1.8 Checkpoint Engine Configuration

| Parameter | Default Value | Description |
|-----------|----------------|-------------|
| `actor_rollout_ref.rollout.checkpoint_engine.backend` | `naive` | Checkpoint engine backend |
| `actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes` | `2048` | Weight update bucket size (MB) |

#### 1.1.9 Trace Configuration

| Parameter Name | Default Value | Description |
|--------|--------|------|
| `actor_rollout_ref.rollout.trace.project_name` | Referenced from `trainer.project_name` | The trace project name |
| `actor_rollout_ref.rollout.trace.experiment_name` | Referenced from `trainer.experiment_name` | The trace experiment name |
| `actor_rollout_ref.rollout.trace.backend` | `null` | The trace backend |
| `actor_rollout_ref.rollout.trace.token2text` | `false` | Whether to convert tokens to text |
| `actor_rollout_ref.rollout.trace.max_samples_per_step_per_worker` | `null` | The maximum number of samples per step per worker |

#### 1.1.10 Prometheus Configuration

| Parameter Name | Default Value | Description |
|--------|--------|------|
| `actor_rollout_ref.rollout.prometheus.enable` | `false` | Whether to enable Prometheus monitoring |
| `actor_rollout_ref.rollout.prometheus.port` | `9090` | Prometheus port |
| `actor_rollout_ref.rollout.prometheus.file` | `/tmp/ray/session_latest/metrics/prometheus/prometheus.yml` | Prometheus configuration file path |
| `actor_rollout_ref.rollout.prometheus.served_model_name` | Referenced from `model.path` | Served model name |

#### 1.1.11 Reference Model Configuration

| Parameter | Default Value | Description |
|--------|--------|------|
| `actor_rollout_ref.ref.rollout_n` | Inherited from `rollout.n` | Number of rollout iterations |
| `actor_rollout_ref.ref.strategy` | Inherited from `actor.strategy` | Training strategy |
| `actor_rollout_ref.ref.use_torch_compile` | Inherited from `actor.use_torch_compile` | Whether to use torch.compile |
| `actor_rollout_ref.ref.log_prob_micro_batch_size` | `null` | Micro batch size for log prob computation |
| `actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu` | `null` | Log prob micro batch size per GPU |
| `actor_rollout_ref.ref.log_prob_use_dynamic_bsz` | Inherited from `actor.use_dynamic_bsz` | Whether to use dynamic batch size for log prob |
| `actor_rollout_ref.ref.log_prob_max_token_len_per_gpu` | Inherited from `actor.ppo_max_token_len_per_gpu` | Maximum token length for log prob per GPU |
| `actor_rollout_ref.ref.ulysses_sequence_parallel_size` | Inherited from `actor.ulysses_sequence_parallel_size` | Ulysses sequence parallel size |
| `actor_rollout_ref.ref.entropy_from_logits_with_chunking` | `false` | Whether to compute entropy from logits using chunking |
| `actor_rollout_ref.ref.entropy_checkpointing` | `false` | Whether to use gradient checkpointing for entropy computation |

#### 1.1.12 Critic Optimizer Configuration

| Parameter Name | Default Value | Description |
|--------|--------|------|
| `critic.optim.lr` | `1.0e-05` | Critic learning rate |
| `critic.optim.lr_warmup_steps_ratio` | `0.0` | Learning rate warmup steps ratio |
| `critic.optim.total_training_steps` | `-1` | Total training steps |
| `critic.optim.weight_decay` | `0.01` | Weight decay |
| `critic.optim.lr_warmup_steps` | `-1` | Learning rate warmup steps |
| `critic.optim.betas` | `[0.9, 0.999]` | Adam optimizer momentum coefficients |
| `critic.optim.clip_grad` | `1.0` | Gradient clipping threshold |
| `critic.optim.override_optimizer_config` | `null` / `{}` | Override optimizer configuration |

#### 1.1.13 Critic Policy Configuration

| Parameter | Default Value | Description |
|--------|--------|------|
| `critic.strategy` | `fsdp` / `megatron` | Training strategy |
| `critic.enable` | `null` | Whether to enable the Critic; `null` means it is determined automatically |
| `critic.ppo_mini_batch_size` | Referenced from `actor.ppo_mini_batch_size` | PPO mini batch size |
| `critic.ppo_micro_batch_size` | `null` | PPO micro batch size |
| `critic.ppo_micro_batch_size_per_gpu` | `null` | PPO micro batch size per GPU |
| `critic.use_dynamic_bsz` | Referenced from `actor.use_dynamic_bsz` | Whether to use a dynamic batch size |
| `critic.ppo_max_token_len_per_gpu` | `32768` | Maximum PPO token length per GPU |
| `critic.forward_max_token_len_per_gpu` | Referenced from `critic.ppo_max_token_len_per_gpu` | Maximum token length per GPU for forward computation |
| `critic.ppo_epochs` | Referenced from `actor.ppo_epochs` | Number of PPO update epochs |
| `critic.shuffle` | Referenced from `actor.shuffle` | Whether to shuffle |
| `critic.data_loader_seed` | `42` / Referenced from `actor.data_loader_seed` | Random seed for the data loader |
| `critic.cliprange_value` | `0.5` | Clipping range for the Critic value function |
| `critic.loss_agg_mode` | Referenced from `actor.loss_agg_mode` | Loss aggregation mode |
| `critic.grad_clip` | `1.0` | Gradient clipping value |
| `critic.ulysses_sequence_parallel_size` | `1` | Ulysses sequence parallel size |
| `critic.forward_micro_batch_size` | Referenced from `critic.ppo_micro_batch_size` | Micro batch size for forward computation |
| `critic.forward_micro_batch_size_per_gpu` | Referenced from `critic.ppo_micro_batch_size_per_gpu` | Micro batch size per GPU for forward computation |

#### 1.1.14 Critic Model Configuration

| Parameter | Default Value | Description |
|-----------|---------------|--------------|
| `critic.model.path` | `~/models/deepseek-llm-7b-chat` | Path to the critic model |
| `critic.model.tokenizer_path` | Referenced from `model.path` | Path to the tokenizer |
| `critic.model.override_config` | `{}` | Overrides the model configuration |
| `critic.model.external_lib` | Referenced from `model.external_lib` | Path to the external library |
| `critic.model.trust_remote_code` | Referenced from `model.trust_remote_code` | Whether to trust remote code |
| `critic.model.use_shm` | `false` | Whether to use shared memory |
| `critic.model.enable_gradient_checkpointing` | `true` | Whether to enable gradient checkpointing |
| `critic.model.enable_activation_offload` | `false` | Whether to enable activation offloading |
| `critic.model.use_remove_padding` | `false` / `true` | Whether to remove padding |
| `critic.model.lora_rank` | `0` | LoRA rank |
| `critic.model.lora_alpha` | `16` | LoRA alpha |
| `critic.model.target_modules` | `all-linear` | Target modules for LoRA |
| `critic.model.tiled_mlp.enabled` | `false` | Whether to enable tiled MLP |
| `critic.model.tiled_mlp.num_shards` | `4` | Number of MLP shards |

#### 1.1.15 Data Configuration

| Parameter | Default Value | Description |
|--------|--------|------|
| `data.tokenizer` | `null` | Path to the tokenizer |
| `data.use_shm` | `false` | Whether to use shared memory |
| `data.train_files` | `~/data/rlhf/gsm8k/train.parquet` | Path to the training data file |
| `data.val_files` | `~/data/rlhf/gsm8k/test.parquet` | Path to the validation data file |
| `data.train_max_samples` | `-1` | Maximum number of training samples; `-1` means no limit |
| `data.val_max_samples` | `-1` | Maximum number of validation samples |
| `data.prompt_key` | `prompt` | Key name for the prompt in the data |
| `data.reward_fn_key` | `data_source` | Key name for the reward function |
| `data.max_prompt_length` | `512` | Maximum prompt length |
| `data.max_response_length` | `512` | Maximum response length |
| `data.train_batch_size` | `1024` | Training batch size |
| `data.val_batch_size` | `null` | Validation batch size |
| `data.tool_config_path` | Referenced from `rollout.multi_turn.tool_config_path` | Path to the tool configuration file |
| `data.return_raw_input_ids` | `false` | Whether to return raw input IDs |
| `data.return_raw_chat` | `true` | Whether to return raw chat content |
| `data.return_full_prompt` | `false` | Whether to return the full prompt |
| `data.shuffle` | `true` | Whether to shuffle the training data |
| `data.seed` | `null` | Random seed for data shuffling |
| `data.dataloader_num_workers` | `8` | Number of worker processes for the data loader |
| `data.image_patch_size` | `14` | Image patch size |
| `data.validation_shuffle` | `false` | Whether to shuffle during validation |
| `data.filter_overlong_prompts` | `false` | Whether to filter overlong prompts |
| `data.filter_overlong_prompts_workers` | `1` | Number of worker processes for filtering overlong prompts |
| `data.truncation` | `error` | Truncation strategy |
| `data.image_key` | `images` | Key name for image data |
| `data.video_key` | `videos` | Key name for video data |
| `data.trust_remote_code` | `false` | Whether to trust remote code |
| `data.return_multi_modal_inputs` | `true` | Whether to return multimodal inputs |

#### 1.1.16 Reward Configuration

| Parameter | Default Value | Description |
|--------|--------|------|
| `reward.num_workers` | `8` | Number of reward computation worker processes |
| `reward.custom_reward_function.path` | `null` | Path to the custom reward function |
| `reward.custom_reward_function.name` | `compute_score` | Name of the custom reward function |
| `reward.reward_manager.source` | `register` | Source of the reward manager |
| `reward.reward_manager.name` | `naive` | Name of the reward manager |
| `reward.reward_model.enable` | `false` | Whether to enable the reward model |
| `reward.reward_model.enable_resource_pool` | `false` | Whether to enable the reward model resource pool |
| `reward.reward_model.n_gpus_per_node` | `8` | Number of GPUs per node for the reward model |
| `reward.reward_model.nnodes` | `0` | Number of nodes for the reward model |
| `reward.reward_model.model_path` | `null` | Path to the reward model |
| `reward.sandbox_fusion.url` | `null` | Sandbox Fusion URL |
| `reward.sandbox_fusion.max_concurrent` | `64` | Maximum number of concurrent Sandbox Fusion operations |
| `reward.sandbox_fusion.memory_limit_mb` | `1024` | Sandbox Fusion memory limit (MB) |

#### 1.1.17 Algorithm Configuration

| Parameter | Default Value | Description |
|--------|--------|------|
| `algorithm.gamma` | `1.0` | Discount factor |
| `algorithm.lam` | `1.0` | GAE lambda parameter |
| `algorithm.adv_estimator` | `gae` | Advantage estimation method, for example, gae |
| `algorithm.norm_adv_by_std_in_grpo` | `true` | Whether to normalize advantages by standard deviation in GRPO |
| `algorithm.use_kl_in_reward` | `false` | Whether to use KL penalty in the reward |
| `algorithm.kl_penalty` | `kl` | KL penalty type |
| `algorithm.kl_ctrl.type` | `fixed` | KL controller type, for example, fixed or kl_adapter |
| `algorithm.kl_ctrl.kl_coef` | `0.001` | KL penalty coefficient |
| `algorithm.kl_ctrl.horizon` | `10000` | Horizon of the KL adapter |
| `algorithm.kl_ctrl.target_kl` | `0.1` | Target KL divergence |
| `algorithm.use_pf_ppo` | `false` | Whether to use PF-PPO |
| `algorithm.pf_ppo.reweight_method` | `pow` | PF-PPO reweighting method |
| `algorithm.pf_ppo.weight_pow` | `2.0` | PF-PPO weighting power |

#### 1.1.18 Rollout Correction Configuration

| Parameter | Default Value | Description |
|--------|--------|------|
| `algorithm.rollout_correction.rollout_is` | `null` | Whether to enable IS importance sampling correction |
| `algorithm.rollout_correction.rollout_is_threshold` | `2.0` | IS weight threshold |
| `algorithm.rollout_correction.rollout_rs` | `null` | Whether to enable rejection sampling correction |
| `algorithm.rollout_correction.rollout_rs_threshold` | `null` | RS threshold |
| `algorithm.rollout_correction.bypass_mode` | `false` | Whether to enable bypass mode |
| `algorithm.rollout_correction.loss_type` | `ppo_clip` | Correction loss type |
| `algorithm.rollout_correction.rollout_is_batch_normalize` | `false` | Whether IS weights are batch-normalized |

#### 1.1.19 Trainer Configuration

| Parameter | Default Value | Description |
|--------|--------|------|
| `trainer.balance_batch` | `true` | Whether to balance the batch |
| `trainer.total_epochs` | `30` | Total number of training epochs |
| `trainer.total_training_steps` | `null` | Total number of training steps; `null` means it is automatically calculated from the epochs |
| `trainer.project_name` | `verl_examples` | Project name |
| `trainer.experiment_name` | `gsm8k` | Experiment name |
| `trainer.logger` | `[console, wandb]` | List of logging backends |
| `trainer.log_val_generations` | `0` | Number of validation generation logs |
| `trainer.nnodes` | `1` | Number of training nodes |
| `trainer.n_gpus_per_node` | `8` | Number of GPUs per node |
| `trainer.save_freq` | `-1` | Saving frequency; `-1` means do not save |
| `trainer.esi_redundant_time` | `0` | ESI redundant time |
| `trainer.resume_mode` | `auto` | Resume mode; options include `auto` and so on |
| `trainer.resume_from_path` | `null` | Resume path |
| `trainer.val_before_train` | `true` | Whether to validate before training |
| `trainer.val_only` | `false` | Whether to validate only |
| `trainer.test_freq` | `-1` | Testing frequency |
| `trainer.critic_warmup` | `0` | Number of Critic warmup steps |
| `trainer.default_hdfs_dir` | `null` | Default HDFS directory |
| `trainer.del_local_ckpt_after_load` | `false` | Whether to delete the local checkpoint after loading |
| `trainer.default_local_dir` | `checkpoints/${trainer.project_name}/${trainer.experiment_name}` | Default local checkpoint directory |
| `trainer.max_actor_ckpt_to_keep` | `null` | Maximum number of Actor checkpoints to keep |
| `trainer.max_critic_ckpt_to_keep` | `null` | Maximum number of Critic checkpoints to keep |
| `trainer.ray_wait_register_center_timeout` | `300` | Ray registration center waiting timeout (seconds) |
| `trainer.device` | `cuda` | Training device |
| `trainer.use_legacy_worker_impl` | `auto` | Whether to use the legacy worker implementation |
| `trainer.rollout_data_dir` | `null` | Address configuration for saving the rollout results of each round |

#### 1.1.20 Model Configuration

| Parameter Name | Default Value | Description |
|--------|--------|------|
| `actor_rollout_ref.model.path` | `~/models/deepseek-llm-7b-chat` | Model path |
| `actor_rollout_ref.model.hf_config_path` | `null` | HuggingFace configuration path |
| `actor_rollout_ref.model.tokenizer_path` | `null` | Tokenizer path |
| `actor_rollout_ref.model.use_shm` | `false` | Whether to use shared memory |
| `actor_rollout_ref.model.trust_remote_code` | `false` | Whether to trust remote code |
| `actor_rollout_ref.model.custom_chat_template` | `null` | Custom chat template |
| `actor_rollout_ref.model.external_lib` | `null` | External library path |
| `actor_rollout_ref.model.override_config` | `{}` | Override model configuration |
| `actor_rollout_ref.model.enable_gradient_checkpointing` | `true` | Whether to enable gradient checkpointing |
| `actor_rollout_ref.model.enable_activation_offload` | `false` | Whether to enable activation offloading |
| `actor_rollout_ref.model.use_remove_padding` | `true` / `false` | Whether to remove padding |
| `actor_rollout_ref.model.lora_rank` | `0` | LoRA rank; 0 means LoRA is not used |
| `actor_rollout_ref.model.lora_alpha` | `16` | LoRA alpha |
| `actor_rollout_ref.model.target_modules` | `all-linear` | LoRA target modules |
| `actor_rollout_ref.model.exclude_modules` | `null` | LoRA excluded modules |
| `actor_rollout_ref.model.lora_adapter_path` | `null` | LoRA adapter path |
| `actor_rollout_ref.model.use_liger` | `false` | Whether to use Liger kernels |
| `actor_rollout_ref.model.use_fused_kernels` | `false` | Whether to use fused kernels |
| `actor_rollout_ref.model.fused_kernel_options.impl_backend` | `torch` | Implementation backend for fused kernels |
| `actor_rollout_ref.model.tiled_mlp.enabled` | `false` | Whether to enable tiled MLP |
| `actor_rollout_ref.model.tiled_mlp.num_shards` | `4` | Number of MLP shards |

#### 1.1.21 Common Engine Configuration

| Parameter | Default Value | Description |
|-----------|----------------|--------------|
| `actor_rollout_ref.hybrid_engine` | `true` | Whether to use the hybrid engine (training and inference share weights) |
| `actor_rollout_ref.nccl_timeout` | `600` | NCCL communication timeout (in seconds) |
| `transfer_queue.enable` | `false` | Whether to enable the transfer queue |

---

### 1.2 FSDP-Specific Configuration Parameters

The following parameters exist only in the FSDP solution (`_generated_ppo_trainer.yaml`).

#### 1.2.1 FSDP Optimizer Configuration

| Parameter Name | Default Value | Description |
|--------|--------|------|
| `actor_rollout_ref.actor.optim.optimizer` | `AdamW` | Optimizer type |
| `actor_rollout_ref.actor.optim.optimizer_impl` | `torch.optim` | Optimizer implementation |
| `actor_rollout_ref.actor.optim.min_lr_ratio` | `0.0` | Minimum learning rate ratio |
| `actor_rollout_ref.actor.optim.num_cycles` | `0.5` | Number of cosine scheduling cycles |
| `actor_rollout_ref.actor.optim.lr_scheduler_type` | `constant` | Learning rate scheduler type |
| `actor_rollout_ref.actor.optim.zero_indexed_step` | `true` | Whether step counting starts from 0 |
| `actor_rollout_ref.actor.optim.warmup_style` | `null` | Warmup style |

#### 1.2.2 Actor FSDP Engine Configuration

| Parameter Name | Default Value | Description |
|--------|--------|------|
| `actor_rollout_ref.actor.fsdp_config.wrap_policy.min_num_params` | `0` | Minimum number of parameters for FSDP wrapping |
| `actor_rollout_ref.actor.fsdp_config.param_offload` | `false` | Whether to offload parameters to the CPU |
| `actor_rollout_ref.actor.fsdp_config.optimizer_offload` | `false` | Whether to offload optimizer states to the CPU |
| `actor_rollout_ref.actor.fsdp_config.offload_policy` | `false` | Offload policy |
| `actor_rollout_ref.actor.fsdp_config.reshard_after_forward` | `true` | Whether to reshard after the forward pass |
| `actor_rollout_ref.actor.fsdp_config.fsdp_size` | `-1` | FSDP group size; `-1` indicates global |
| `actor_rollout_ref.actor.fsdp_config.forward_prefetch` | `false` | Whether to prefetch forward parameters |
| `actor_rollout_ref.actor.fsdp_config.model_dtype` | `fp32` | Data type for model computation |
| `actor_rollout_ref.actor.fsdp_config.use_orig_params` | `false` | Whether to use original parameters |
| `actor_rollout_ref.actor.fsdp_config.seed` | `42` | Random seed |
| `actor_rollout_ref.actor.fsdp_config.full_determinism` | `false` | Whether to enable full determinism |
| `actor_rollout_ref.actor.fsdp_config.forward_only` | `false` | Whether to perform forward computation only (`false` for Actor) |
| `actor_rollout_ref.actor.fsdp_config.strategy` | `fsdp` | Strategy type |
| `actor_rollout_ref.actor.fsdp_config.dtype` | `bfloat16` | Data type for model storage |

#### 1.2.3 Reference FSDP Engine Configuration

The configuration structure is the same as that of the Actor FSDP engine, with the following key differences:

| Parameter Name | Default Value | Description |
|--------|--------|------|
| `actor_rollout_ref.ref.fsdp_config.forward_only` | `true` | The reference model performs forward computation only |

The remaining parameters (`wrap_policy`, `param_offload`, `optimizer_offload`, `reshard_after_forward`, `fsdp_size`, `dtype`, and so on) have default values that are consistent with the Actor FSDP engine configuration.

#### 1.2.4 Critic FSDP Engine Configuration

The configuration structure is the same as that of the Actor FSDP engine, with the following key differences:

| Parameter | Default Value | Description |
|-----------|----------------|--------------|
| `critic.model.fsdp_config.forward_only` | `false` | The critic model requires training |
| `critic.model.fsdp_config.use_remove_padding` | `false` | The critic does not remove padding |

The remaining parameters have the same default values as the Actor FSDP engine configuration.

---

### 1.3 Megatron-Specific Configuration Parameters

The following parameters exist only in the Megatron solution (`_generated_ppo_megatron_trainer.yaml`).

#### 1.3.1 Megatron Optimizer Configuration

| Parameter Name | Default Value | Description |
|--------|--------|------|
| `actor_rollout_ref.actor.optim.optimizer` | `adam` | Optimizer type |
| `actor_rollout_ref.actor.optim.lr_warmup_init` | `0.0` | Initial value for learning rate warmup |
| `actor_rollout_ref.actor.optim.lr_decay_steps` | `null` | Number of steps for learning rate decay |
| `actor_rollout_ref.actor.optim.lr_decay_style` | `constant` | Learning rate decay style, options include constant, cosine, exponential, and so on |
| `actor_rollout_ref.actor.optim.min_lr` | `0.0` | Minimum learning rate |
| `actor_rollout_ref.actor.optim.weight_decay_incr_style` | `constant` | Weight decay increase style |
| `actor_rollout_ref.actor.optim.lr_wsd_decay_style` | `exponential` | WSD learning rate decay style |
| `actor_rollout_ref.actor.optim.lr_wsd_decay_steps` | `null` | Number of steps for WSD learning rate decay |
| `actor_rollout_ref.actor.optim.use_checkpoint_opt_param_scheduler` | `false` | Whether to use the checkpoint optimizer parameter scheduler |

#### 1.3.2 Actor Megatron Engine Configuration

| Parameter Name | Default Value | Description |
|--------|--------|------|
| `actor_rollout_ref.actor.megatron.param_offload` | `false` | Whether to offload parameters to the CPU and release gradient buffers when the actor is inactive |
| `actor_rollout_ref.actor.megatron.optimizer_offload` | `false` | Whether to offload optimizer states to the CPU |
| `actor_rollout_ref.actor.megatron.tensor_model_parallel_size` | `1` | Tensor parallelism (TP) size |
| `actor_rollout_ref.actor.megatron.expert_model_parallel_size` | `1` | Expert parallelism size |
| `actor_rollout_ref.actor.megatron.expert_tensor_parallel_size` | `null` | Expert tensor parallelism (TP) size |
| `actor_rollout_ref.actor.megatron.pipeline_model_parallel_size` | `1` | Pipeline parallelism (PP) size |
| `actor_rollout_ref.actor.megatron.virtual_pipeline_model_parallel_size` | `null` | Virtual pipeline parallelism (PP) size |
| `actor_rollout_ref.actor.megatron.context_parallel_size` | `1` | Context parallelism size |
| `actor_rollout_ref.actor.megatron.sequence_parallel` | `true` | Whether to enable sequence parallelism |
| `actor_rollout_ref.actor.megatron.use_distributed_optimizer` | `true` | Whether to use the distributed optimizer |
| `actor_rollout_ref.actor.megatron.use_dist_checkpointing` | `false` | Whether to use distributed checkpointing |
| `actor_rollout_ref.actor.megatron.dist_checkpointing_path` | `null` | Distributed checkpointing path |
| `actor_rollout_ref.actor.megatron.dist_checkpointing_prefix` | `''` | Distributed checkpointing prefix |
| `actor_rollout_ref.actor.megatron.dist_ckpt_optim_fully_reshardable` | `false` | Whether the distributed checkpointing optimizer is fully reshardable |
| `actor_rollout_ref.actor.megatron.distrib_optim_fully_reshardable_mem_efficient` | `false` | Whether distributed optimizer resharding is memory-efficient |
| `actor_rollout_ref.actor.megatron.seed` | `42` | Random seed |
| `actor_rollout_ref.actor.megatron.use_mbridge` | `true` | Whether to enable Bridge weight conversion |
| `actor_rollout_ref.actor.megatron.vanilla_mbridge` | `false` | Whether to use the deprecated legacy mBridge; Megatron-Bridge is used by default |
| `actor_rollout_ref.actor.megatron.use_remove_padding` | `true` | Whether to remove padding |
| `actor_rollout_ref.actor.megatron.forward_only` | `false` | Whether to perform forward computation only |
| `actor_rollout_ref.actor.megatron.dtype` | `bfloat16` | Model data type |
| `actor_rollout_ref.actor.megatron.load_weight` | `true` | Whether to load weights |

#### 1.3.3 Megatron Transformer Override Configuration

| Parameter Name | Default Value | Description |
|--------|--------|------|
| `override_transformer_config.recompute_granularity` | `null` | Recompute granularity |
| `override_transformer_config.recompute_modules` | `[core_attn]` | List of recompute modules |
| `override_transformer_config.recompute_method` | `null` | Recompute method |
| `override_transformer_config.recompute_num_layers` | `null` | Number of recompute layers |
| `override_transformer_config.attention_backend` | `flash` | Attention backend |

#### 1.3.4 Reference Megatron Engine Configuration

The configuration structure is the same as that of the Actor Megatron engine, with the main differences being:

| Parameter | Default Value | Description |
|-----------|----------------|-------------|
| `actor_rollout_ref.ref.megatron.forward_only` | `true` | The reference model performs forward computation only. |

The remaining parameters use default values from the Actor Megatron engine configuration (such as `param_offload`, `tensor_model_parallel_size`, and so on).

#### 1.3.5 Critic Megatron Engine Configuration

The configuration structure is the same as that of the Actor Megatron engine, with the main differences as follows:

| Parameter | Default Value | Description |
|-----------|----------------|--------------|
| `critic.megatron.forward_only` | `false` | The Critic model requires training |

#### 1.3.6 Megatron LoRA Configuration

| Parameter | Default Value | Description |
|--------|--------|------|
| `model.lora.type` | `lora` | LoRA type |
| `model.lora.merge` | `false` | Whether to merge LoRA weights |
| `model.lora.rank` | `0` | LoRA rank, `0` means not used |
| `model.lora.alpha` | `32` | LoRA alpha |
| `model.lora.dropout` | `0.0` | LoRA dropout |
| `model.lora.target_modules` | `[linear_qkv, linear_proj, linear_fc1, linear_fc2]` | LoRA target modules |
| `model.lora.exclude_modules` | `[]` | LoRA excluded modules |
| `model.lora.dropout_position` | `pre` | LoRA dropout position |
| `model.lora.lora_A_init_method` | `xavier` | Initialization method for LoRA A matrix |
| `model.lora.lora_B_init_method` | `zero` | Initialization method for LoRA B matrix |
| `model.lora.a2a_experimental` | `false` | Whether to enable the a2a experimental feature |
| `model.lora.dtype` | `null` | LoRA data type |
| `model.lora.adapter_path` | `null` | LoRA adapter path |
| `model.lora.freeze_vision_model` | `true` | Whether to freeze the vision model |
| `model.lora.freeze_vision_projection` | `true` | Whether to freeze the vision projection |
| `model.lora.freeze_language_model` | `true` | Whether to freeze the language model |

#### 1.3.7 Model override_config (Megatron Solution)

| Parameter Name | Default Value | Description |
|----------------|---------------|-------------|
| `model.override_config.model_config` | `{}` | Model configuration override |
| `model.override_config.moe_config.freeze_moe_router` | `false` | Whether to freeze the MoE router |

#### 1.3.8 Rollout layer_name_map (Megatron Approach)

| Parameter Name | Default Value | Description |
|----------------|----------------|-------------|
| `rollout.layer_name_map.qkv_layer_name` | `qkv` | QKV layer name mapping |
| `rollout.layer_name_map.gate_proj_layer_name` | `gate_up` | Gate projection layer name mapping |

---

### 1.4 Advanced Configuration Parameters

#### 1.4.1 Profiler Configuration

| Parameter | Default Value | Description |
|-----------|----------------|-------------|
| `profiler.enable` | `false` | Whether to enable the Profiler |
| `profiler.tool` | Referenced from `global_profiler.tool` | Profiler tool. Options: nsys, npu, torch, torch_memory |
| `profiler.all_ranks` | `false` | Whether to enable it on all ranks |
| `profiler.ranks` | `[]` | Specifies the list of ranks to enable |
| `profiler.save_path` | Referenced from `global_profiler.save_path` | Path for saving Profiler results |

#### 1.4.2 Global Profiler Configuration

| Parameter | Default Value | Description |
|-----------|----------------|-------------|
| `global_profiler.tool` | `null` | Global Profiler tool |
| `global_profiler.steps` | `null` | Number of steps for Profiler collection |
| `global_profiler.profile_continuous_steps` | `false` | Whether to collect continuous steps |
| `global_profiler.save_path` | `outputs/profile` | Global save path |

#### 1.4.3 Router Replay Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `actor.megatron.router_replay.mode` / `actor.veomni.router_replay.mode` | `disabled` | Engine-side router replay mode. Options: `disabled`, `R2`, `R3`. |
| `router_replay.record_file` | `null` | Path to the router record file. |
| `router_replay.replay_file` | `null` | Path to the router replay file. |

#### 1.4.4 Checkpoint Configuration

| Parameter | Default Value | Description |
|-----------|----------------|-------------|
| `checkpoint.save_contents` | `[model, optimizer, extra]` | Contents to save in the checkpoint |
| `checkpoint.load_contents` | Inherited from `checkpoint.save_contents` | Contents to load from the checkpoint |
| `checkpoint.async_save` | `false` | Whether to save the checkpoint asynchronously |
| `checkpoint.mbridge_config` | `{}` | mBridge configuration |

#### 1.4.5 QAT Configuration

| Parameter | Default Value | Description |
|-----------|----------------|--------------|
| `qat.enable` | `false` | Whether to enable quantization-aware training (QAT) |
| `qat.mode` | `w4a16` | Quantization mode |
| `qat.group_size` | `16` | Quantization group size |
| `qat.ignore_patterns` | `[lm_head, embed_tokens, re:.*mlp.gate$]` | List of patterns to ignore during quantization |
| `qat.activation_observer` | `static_minmax` | Activation observer type |
| `qat.quantization_config_path` | `null` | Path to the quantization configuration file |

#### 1.4.6 MTP Configuration

| Parameter | Default Value | Description |
|-----------|---------------|--------------|
| `mtp.enable` | `false` | Whether to enable multi-token prediction (MTP) |
| `mtp.enable_train` | `false` | Whether to enable MTP during training |
| `mtp.enable_rollout` | `false` | Whether to enable MTP during inference |
| `mtp.detach_encoder` | `false` | Whether to detach the encoder |
| `mtp.mtp_loss_scaling_factor` | `0.1` | MTP loss scaling factor |
| `mtp.speculative_algorithm` | `EAGLE` | Speculative decoding algorithm |
| `mtp.speculative_num_steps` | `3` | Number of speculative steps |
| `mtp.speculative_eagle_topk` | `1` | EAGLE Top-K |
| `mtp.speculative_num_draft_tokens` | `4` | Number of speculative draft tokens |
| `mtp.method` | `mtp` | MTP method |
| `mtp.num_speculative_tokens` | `1` | Number of speculative tokens |

---

## 2. Training Metrics Description

The log metrics printed by the reinforcement learning algorithm at each iteration are described as follows:

### 2.1 Basic Training Metrics

| Metric | Description |
|--------|-------------|
| `training/global_step` | The current global training step |
| `training/epoch` | The current training epoch |

### 2.2 Actor Model Metrics

| Metric | Description |
|------|------|
| `actor/pg_loss` | Policy gradient loss (PPO clip loss), the value of the policy gradient objective function based on the advantage function |
| `actor/kl_loss` | KL divergence loss, which measures the deviation between the current policy and the reference policy (printed only when `use_kl_loss=True`) |
| `actor/entropy` | Policy entropy, which indicates the randomness or exploration capability of the policy (printed only when `calculate_entropy=True` or `entropy_coeff!=0`) |
| `actor/grad_norm` | Actor gradient norm (after clipping), which indicates the overall magnitude of parameter gradients during backpropagation |
| `actor/lr` | Current learning rate of the Actor |
| `actor/pg_clipfrac` | Proportion of PPO clipping mechanism in effect, which reflects the stability of policy update magnitude |
| `actor/ppo_kl` | Actual KL divergence of the PPO algorithm (current policy vs. old policy) |
| `actor/pg_clipfrac_lower` | Lower-bound clipping proportion of PPO (available for some `loss_mode` values) |
| `actor/reward_kl_penalty` | KL penalty value, the mean KL divergence between the current policy and the reference policy (printed only when `use_kl_in_reward=True`) |
| `actor/reward_kl_penalty_coeff` | KL penalty coefficient beta (printed only when `use_kl_in_reward=True`) |
| `actor/kl_coef` | KL loss coefficient (printed only when `use_kl_loss=True`) |

### 2.3 Critic Model Metrics

| Metric | Description |
|--------|-------------|
| `critic/vf_loss` | Value function loss |
| `critic/vf_clipfrac` | Proportion of steps where the Critic clipping mechanism takes effect, reflecting the stability of value function updates |
| `critic/vpred_mean` | Mean of predicted values |
| `critic/grad_norm` | Critic gradient norm (after clipping) |
| `critic/lr` | Current learning rate of the Critic |
| `critic/vf_explained_var` | Explained variance of the value function, 1 - Var(returns-values)/Var(returns) (printed only when `use_critic=True`) |

### 2.4 Data Statistics Metrics

| Metric | Description |
|------|------|
| `critic/score/mean` | Mean sequence score for non-aborted samples |
| `critic/score/max` | Maximum sequence score for non-aborted samples |
| `critic/score/min` | Minimum sequence score for non-aborted samples |
| `critic/rewards/mean` | Mean sequence reward for non-aborted samples |
| `critic/rewards/max` | Maximum sequence reward for non-aborted samples |
| `critic/rewards/min` | Minimum sequence reward for non-aborted samples |
| `critic/advantages/mean` | Mean advantage value for valid tokens |
| `critic/advantages/max` | Maximum advantage value for valid tokens |
| `critic/advantages/min` | Minimum advantage value for valid tokens |
| `critic/returns/mean` | Mean return for valid tokens |
| `critic/returns/max` | Maximum return for valid tokens |
| `critic/returns/min` | Minimum return for valid tokens |
| `critic/values/mean` | Mean Critic value for valid tokens (printed only when `use_critic=True`) |
| `critic/values/max` | Maximum Critic value for valid tokens (printed only when `use_critic=True`) |
| `critic/values/min` | Minimum Critic value for valid tokens (printed only when `use_critic=True`) |
| `response_length/mean` | Mean response length (including aborted samples) |
| `response_length/max` | Maximum response length |
| `response_length/min` | Minimum response length |
| `response_length/clip_ratio` | Proportion of response lengths reaching the maximum length |
| `response_length_non_aborted/mean` | Mean response length for non-aborted samples |
| `response_length_non_aborted/max` | Maximum response length for non-aborted samples |
| `response_length_non_aborted/min` | Minimum response length for non-aborted samples |
| `response_length_non_aborted/clip_ratio` | Proportion of non-aborted samples whose response length reaches the maximum length |
| `response/aborted_ratio` | Proportion of aborted samples (response length is 0) |
| `prompt_length/mean` | Mean prompt length |
| `prompt_length/max` | Maximum prompt length |
| `prompt_length/min` | Minimum prompt length |
| `prompt_length/clip_ratio` | Proportion of prompt lengths reaching the maximum length |
| `num_turns/mean` | Mean number of turns in multi-turn dialogue (printed only for multi-turn dialogue) |
| `num_turns/max` | Maximum number of turns in multi-turn dialogue (printed only for multi-turn dialogue) |
| `num_turns/min` | Minimum number of turns in multi-turn dialogue (printed only for multi-turn dialogue) |
| `tool_call_counts/mean` | Mean number of tool calls (printed only when `tool_call_counts` exists) |
| `tool_call_counts/max` | Maximum number of tool calls |
| `tool_call_counts/min` | Minimum number of tool calls |

### 2.5 Time Metrics

| Metric | Description |
|------|------|
| `timing_s/gen` | Time spent on generation (rollout) in seconds |
| `timing_s/ref` | Time spent by the Reference model computing log_p in seconds |
| `timing_s/values` | Time spent by the Critic model computing values in seconds |
| `timing_s/adv` | Time spent computing advantage values in seconds |
| `timing_s/update_critic` | Time spent updating the Critic model in seconds |
| `timing_s/update_actor` | Time spent updating the Actor model in seconds |
| `timing_s/step` | Total time spent on one step in seconds |
| `timing_s/old_log_prob` | Time spent by the Actor model computing old log_p in seconds |
| `timing_s/reward` | Time spent computing rewards in seconds |
| `timing_s/testing` | Time spent on validation in seconds |
| `timing_s/save_checkpoint` | Time spent saving checkpoints in seconds |
| `timing_s/update_weights` | Time spent synchronizing weights in seconds |
| `timing_per_token_ms/gen` | Time spent per token during the generation phase in milliseconds |
| `timing_per_token_ms/ref` | Time spent per token by the Reference model in milliseconds |
| `timing_per_token_ms/values` | Time spent per token by the Critic model in milliseconds |
| `timing_per_token_ms/adv` | Time spent per token computing advantage values in milliseconds |
| `timing_per_token_ms/update_critic` | Time spent per token updating the Critic model in milliseconds |
| `timing_per_token_ms/update_actor` | Time spent per token updating the Actor model in milliseconds |

### 2.6 Performance Metrics

| Metric | Description |
|------|------|
| `perf/total_num_tokens` | Total number of tokens processed in this step |
| `perf/time_per_step` | Total time spent in this step (seconds) |
| `perf/throughput` | Throughput: tokens / (time * n_gpus) |
| `perf/max_memory_allocated_gb` | Maximum allocated GPU memory (GB) |
| `perf/max_memory_reserved_gb` | Maximum reserved GPU memory (GB) |
| `perf/cpu_memory_used_gb` | CPU memory used (GB) |
| `perf/mfu/actor` | MFU (model floating-point utilization) for Actor training |
| `perf/mfu/critic` | MFU for Critic training |
| `perf/mfu/actor_infer` | MFU for the Actor inference phase |

### 2.7 Variance Proxy Metric

| Metric | Description |
|------|------|
| `variance_proxy/proxy1_signal_strength` | Signal strength: the squared norm of the gradient mean \|\|g_mean\|\|^2 |
| `variance_proxy/proxy2_total_power` | Total power: the expected value of the squared norm of the gradient E[\|\|g_tau\|\|^2] |
| `variance_proxy/proxy3_pure_noise` | Pure noise: the gradient variance proxy (1/(N-1)) * (Proxy2 - Proxy1) |
| `variance_proxy/expected_a_squared` | The expected value of the squared advantage E[A^2] |
| `variance_proxy/expected_w` | The expected value of the W-score proxy E[W] |

### 2.8 Conditional Metrics

The following metrics are printed only when specific conditions are met:

#### 2.8.1 Rollout Correction Metrics

Printed only when `rollout_correction` is enabled, all with the `rollout_corr/` prefix.

**IS weight metric** (only when IS correction is enabled):

| Metric | Description |
|------|------|
| `rollout_corr/rollout_is_mean` | Mean of IS weights |
| `rollout_corr/rollout_is_max` | Maximum of IS weights |
| `rollout_corr/rollout_is_min` | Minimum of IS weights |
| `rollout_corr/rollout_is_std` | Standard deviation of IS weights |
| `rollout_corr/rollout_is_ratio_fraction_high` | Proportion of IS weights exceeding the upper threshold |
| `rollout_corr/rollout_is_ratio_fraction_low` | Proportion of IS weights below the lower threshold |
| `rollout_corr/rollout_is_eff_sample_size` | Effective sample size (ESS) |
| `rollout_corr/rollout_is_seq_mean` | Mean of sequence-level IS weights |
| `rollout_corr/rollout_is_seq_std` | Standard deviation of sequence-level IS weights |
| `rollout_corr/rollout_is_seq_max` | Maximum of sequence-level IS weights |
| `rollout_corr/rollout_is_seq_min` | Minimum of sequence-level IS weights |
| `rollout_corr/rollout_is_seq_max_deviation` | Maximum deviation of sequence-level IS weights from the ideal value of 1.0 |
| `rollout_corr/rollout_is_seq_fraction_high` | Proportion of sequence-level IS weights exceeding the upper threshold |
| `rollout_corr/rollout_is_seq_fraction_low` | Proportion of sequence-level IS weights below the lower threshold |
| `rollout_corr/rollout_is_batch_norm_factor` | Batch normalization factor for IS weights (printed only when `rollout_is_batch_normalize=True`) |

**Rejection Sampling metrics** (only when RS correction is enabled):

| Metric | Description |
|------|------|
| `rollout_corr/rollout_rs_{option}_mean` | Mean of the RS statistic |
| `rollout_corr/rollout_rs_{option}_max` | Maximum value of the RS statistic |
| `rollout_corr/rollout_rs_{option}_min` | Minimum value of the RS statistic |
| `rollout_corr/rollout_rs_{option}_std` | Standard deviation of the RS statistic |
| `rollout_corr/rollout_rs_{option}_fraction_high` | Fraction exceeding the upper threshold |
| `rollout_corr/rollout_rs_{option}_fraction_low` | Fraction below the lower threshold |
| `rollout_corr/rollout_rs_{option}_seq_mean` | Mean of the sequence-level RS statistic |
| `rollout_corr/rollout_rs_{option}_seq_std` | Standard deviation of the sequence-level RS statistic |
| `rollout_corr/rollout_rs_{option}_seq_max` | Maximum value of the sequence-level RS statistic |
| `rollout_corr/rollout_rs_{option}_seq_min` | Minimum value of the sequence-level RS statistic |
| `rollout_corr/rollout_rs_{option}_seq_max_deviation` | Maximum deviation of the sequence-level RS statistic from 0 |
| `rollout_corr/rollout_rs_{option}_seq_fraction_high` | Fraction exceeding the upper limit at the sequence level |
| `rollout_corr/rollout_rs_{option}_seq_fraction_low` | Fraction below the lower limit at the sequence level |
| `rollout_corr/rollout_rs_{option}_masked_fraction` | Fraction masked at the token level |
| `rollout_corr/rollout_rs_{option}_seq_masked_fraction` | Fraction masked at the sequence level |
| `rollout_corr/rollout_rs_masked_fraction` | Overall fraction masked at the token level |
| `rollout_corr/rollout_rs_seq_masked_fraction` | Overall fraction masked at the sequence level |

**Off-policy diagnostic metrics** (only when off-policy diagnostics are enabled):

| Metric | Description |
|------|------|
| `rollout_corr/training_ppl` | Perplexity of the training policy |
| `rollout_corr/training_log_ppl` | Log perplexity of the training policy |
| `rollout_corr/kl` | Direct estimate of KL(π_rollout \|\| π_training) |
| `rollout_corr/k3_kl` | K3 KL estimate (more stable) |
| `rollout_corr/rollout_ppl` | Perplexity of the rollout policy |
| `rollout_corr/rollout_log_ppl` | Log perplexity of the rollout policy |
| `rollout_corr/log_ppl_diff` | Log PPL difference (rollout - training) |
| `rollout_corr/log_ppl_abs_diff` | Mean of the absolute log PPL difference |
| `rollout_corr/log_ppl_diff_max` | Maximum log PPL difference |
| `rollout_corr/log_ppl_diff_min` | Minimum log PPL difference |
| `rollout_corr/ppl_ratio` | PPL ratio (training_ppl / rollout_ppl) |
| `rollout_corr/chi2_token` | Token-level chi-square divergence |
| `rollout_corr/chi2_seq` | Sequence-level chi-square divergence |

#### 2.8.2 Sequence Length Balance Metrics

Print only when `balance_batch` is enabled:

| Metric | Description |
|------|------|
| `global_seqlen/min` | The minimum sum of sequence lengths across DP partitions before balancing |
| `global_seqlen/max` | The maximum sum of sequence lengths across DP partitions before balancing |
| `global_seqlen/minmax_diff` | The max - min difference before balancing |
| `global_seqlen/balanced_min` | The minimum sum of sequence lengths across DP partitions after balancing |
| `global_seqlen/balanced_max` | The maximum sum of sequence lengths across DP partitions after balancing |
| `global_seqlen/mean` | The average sum of sequence lengths across all partitions |

#### 2.8.3 GDPO Reward Metrics

When using only the GDPO estimator, print the following:

| Metric | Description |
|--------|-------------|
| `gdpo/{key}/mean` | Mean of each reward component in GDPO |
| `gdpo/{key}/std` | Standard deviation of each reward component in GDPO |
| `gdpo/{key}/max` | Maximum value of each reward component in GDPO |
| `gdpo/{key}/min` | Minimum value of each reward component in GDPO |

#### 2.8.4 Training and Inference Consistency Metrics

Only when `actor_rollout_ref.rollout.calculate_log_probs=True` is set, the following is printed:

| Metric | Description |
|------|------|
| `training/rollout_probs_diff_valid` | Marked as 1 (valid) |
| `training/rollout_probs_diff_max` | Maximum value of the probability difference between rollout and actor |
| `training/rollout_probs_diff_mean` | Mean value of the probability difference between rollout and actor |
| `training/rollout_probs_diff_std` | Standard deviation of the probability difference between rollout and actor |
| `training/rollout_actor_probs_pearson_corr` | Pearson correlation coefficient between rollout and actor probabilities |

#### 2.8.5 Validation Metrics

During the verification phase, the following is printed:

| Metric | Description |
|------|------|
| `val-core/{data_source}/{var_name}/{metric_name}` | Core validation metrics (mean@N, maj@N, best@N, and so on) |
| `val-aux/{data_source}/{var_name}/{metric_name}` | Auxiliary validation metrics (std@N, worst@N, and so on) |
| `val-aux/num_turns/mean` | Mean number of multi-turn dialogue turns in the validation set |
| `val-aux/num_turns/max` | Maximum number of multi-turn dialogue turns in the validation set |
| `val-aux/num_turns/min` | Minimum number of multi-turn dialogue turns in the validation set |
