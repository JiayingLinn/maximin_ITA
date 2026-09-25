"""Model-independent pairwise training constants."""

ALLOWED_SPLITS = ("train", "validation")
REQUIRED_COLUMNS = {"domain", "history", "human_ref_A", "human_ref_B", "labels"}
MODEL_COLUMNS = {"input_ids_chosen", "attention_mask_chosen", "input_ids_rejected", "attention_mask_rejected"}
DEFAULT_LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")
