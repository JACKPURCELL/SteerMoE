# Copyright 2022 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.


import torch
import pandas as pd

def register_vllm_models():
    from vllm import ModelRegistry
    ModelRegistry.register_model(
        "Qwen3MoeForCausalLM",
        "src.modeling_vllm.qwen3_moe:Qwen3MoeForCausalLM"
    )
    ModelRegistry.register_model(
        "MixtralForCausalLM",
        "src.modeling_vllm.mixtral:MixtralForCausalLM"
    )
    ModelRegistry.register_model(
        "OlmoeForCausalLM",
        "src.modeling_vllm.olmoe:OlmoeForCausalLM"
    )
    ModelRegistry.register_model(
        "Llama4ForConditionalGeneration",
        "src.modeling_vllm.mllama4:Llama4ForConditionalGeneration"
    )
    ModelRegistry.register_model(
        "GptOssForCausalLM",
        "src.modeling_vllm.gpt_oss:GptOssForCausalLM"
    )
    ModelRegistry.register_model(
        "PhiMoEForCausalLM",
        "src.modeling_vllm.phimoe:PhiMoEForCausalLM"
    )


def register_vllm_save_models():
    from vllm import ModelRegistry
    ModelRegistry.register_model(
        "Qwen3MoeForCausalLM",
        "src.modeling_vllm_save.qwen3_moe:Qwen3MoeForCausalLM"
    )
    ModelRegistry.register_model(
        "MixtralForCausalLM",
        "src.modeling_vllm_save.mixtral:MixtralForCausalLM"
    )
    ModelRegistry.register_model(
        "OlmoeForCausalLM",
        "src.modeling_vllm_save.olmoe:OlmoeForCausalLM"
    )
    ModelRegistry.register_model(
        "DeepseekV2ForCausalLM",
        "src.modeling_vllm_save.deepseek_v2:DeepseekV2ForCausalLM"
    )
    ModelRegistry.register_model(
        "Llama4ForConditionalGeneration",
        "src.modeling_vllm_save.mllama4:Llama4ForConditionalGeneration"
    )
    ModelRegistry.register_model(
        "GptOssForCausalLM",
        "src.modeling_vllm_save.gpt_oss:GptOssForCausalLM"
    )
    ModelRegistry.register_model(
        "PhiMoEForCausalLM",
        "src.modeling_vllm_save.phimoe:PhiMoEForCausalLM"
    )


def steer_moe(llm, activations_path, num_pos_experts, num_neg_experts, reverse_effect: bool, strategy: str, steering_magnitude=1000):
    """
    Steer the MoE model based on activation patterns.

    Args:
        llm (LLM): The LLM instance to steer.
        activations_path (str): Path to the activation patterns DataFrame.
        num_experts (int): Number of experts to steer.
        reverse_effect (bool): Whether to reverse the effect of steering.
        strategy (str): Strategy to use for sorting activations for steering.
        steering_magnitude (float): Magnitude of steering. (Deprecated, we use max, min for the magnitude)
        
    Returns:
        str: A string representation of the steering weights.
    """
    activations_df = pd.read_pickle(activations_path)
    num_layers, llm_num_experts, num_model_experts = activations_df["layer"].nunique(), activations_df["expert"].nunique(), len(activations_df)

    if "risk_diff" in strategy:
        activations_df = activations_df.sort_values(by="risk_diff_abs", ascending=False).reset_index(drop=True)
        metric = "risk_diff"
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    ### Reverse
    if reverse_effect:
        activations_df[metric] = -activations_df[metric]

    ### BY TOP EXPERTS
    ### Top experts (num_pos_experts + num_neg_experts)
    print("MAX EXPERTS:", len(activations_df[activations_df[metric] > 0]), len(activations_df[activations_df[metric] < 0]))
    pos_activations_df = activations_df[activations_df[metric] > 0].head(num_pos_experts)
    neg_activations_df = activations_df[activations_df[metric] < 0].head(num_neg_experts)
    activations_df = pd.concat([pos_activations_df, neg_activations_df], ignore_index=True)
    if len(activations_df) != (num_pos_experts + num_neg_experts):
        raise ValueError(f"Expected {num_pos_experts}+{num_neg_experts}={num_pos_experts + num_neg_experts} experts, but got {len(activations_df)}. There are {len(pos_activations_df)} positive and {len(neg_activations_df)} negative experts.")

    print(f"##### Total Experts: {num_model_experts}, Layers: {num_layers}, Experts: {llm_num_experts}")
    print(f"##### Num Experts: {len(activations_df)}, Steering Magnitude: {steering_magnitude}, Reverse Effect: {reverse_effect}, pos_num_experts: {activations_df[activations_df[metric] > 0].shape[0]}, neg_num_experts: {activations_df[activations_df[metric] < 0].shape[0]}, metric={metric}, strategy: {strategy}")

    def set_moe(self):
        model = self.model_runner.model
        model = model.language_model if hasattr(model, "language_model") else model  # For Llama4
        moe_manual_weights = torch.zeros((num_layers, llm_num_experts), device="cuda")
        updated = 0
        for row in activations_df.to_dict(orient="records"):
            if row[metric] > 0:
                num_experts_per_tok = model.config.num_experts_per_tok if hasattr(model, "config") else model.model_config.num_experts_per_tok
                if (moe_manual_weights[row["layer"]] > 0).sum() < num_experts_per_tok:  # add if less than num active experts are already set to 1
                    updated += 1
                    moe_manual_weights[row["layer"]][row["expert"]] = 1
            else:
                moe_manual_weights[row["layer"]][row["expert"]] = -1
                updated += 1

        model.update_moe_manual_args(moe_manual_args={"moe_manual_weights": moe_manual_weights.clone(), "moe_change_scale": steering_magnitude})
        return ""
        # return f"{updated}: " + repr((moe_manual_weights[0, :10].cpu().numpy() * steering_magnitude).tolist())

    print("\n".join(llm.collective_rpc(set_moe)))
    return activations_df

