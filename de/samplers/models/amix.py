import torch
from pathlib import Path
import hydra
from omegaconf import OmegaConf
import sys
from transformers import EsmTokenizer, BatchEncoding
from typing import List


class AMix(torch.nn.Module):
    def __init__(self, ckpt_path: str):
        """
        AMix model wrapper for directed evolution.
        
        Args:
            ckpt_path (str): Path to AMix checkpoint file.
        """
        super(AMix, self).__init__()
        assert ckpt_path is not None, "ckpt_path must be provided"
        
        # Load AMix model
        root_path = Path(ckpt_path).parents[1]
        sys.path.append(str(root_path))
        cfg_path = Path(root_path, ".hydra", "config.yaml")
        
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config file not found at {cfg_path}")
        
        ckpt_cfg = OmegaConf.load(cfg_path)
        
        # Set attention implementation
        ckpt_cfg.model.bfn.net.config._attn_implementation = 'sdpa'
        
        # Instantiate and load model
        self.model = hydra.utils.instantiate(ckpt_cfg.model)
        state_dict = torch.load(ckpt_path, map_location='cpu')['state_dict']
        self.model.load_state_dict(state_dict)
        del state_dict
        
        # Initialize tokenizer (AMix uses same tokenizer as ESM2)
        self.tokenizer = EsmTokenizer.from_pretrained('facebook/esm2_t30_150M_UR50D')
        
    def tokenize(self, inputs: List[str]) -> BatchEncoding:
        """Convert inputs to a format suitable for the model.

        Args:
            inputs (List[str]): A list of protein sequence strings of len [population].

        Returns:
            encoded_inputs (BatchEncoding): a BatchEncoding object.
        """
        encoded_inputs = self.tokenizer(inputs,
                                        add_special_tokens=True,
                                        return_tensors="pt",
                                        padding=True)
        return encoded_inputs

    def decode(self, tokens: torch.Tensor) -> List[str]:
        """Decode predicted tokens into alphabet characters

        Args:
            tokens (torch.Tensor): Predicted tokens of shape [batch, sequence_length]

        Returns:
            (List[str]): Predicted characters.
        """
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)

    def forward(self, inputs: BatchEncoding) -> torch.Tensor:
        """Forward pass of AMix model

        Args:
            inputs (BatchEncoding): Output of tokenizer.

        Returns:
            results: Model outputs with logits and hidden states.
        """
        # AMix expects inputs_embeds, so we need to convert input_ids to embeddings
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)
        
        # Create one-hot encoding for BFN input
        import torch.nn.functional as F
        inputs_embeds = F.one_hot(input_ids, num_classes=len(self.tokenizer)).float()
        
        # Set timestep to 1.0 for inference (fully denoised)
        t = torch.ones_like(attention_mask).float()
        
        # Forward pass through the BFN model
        with torch.no_grad():
            outputs = self.model.bfn.net(
                t=t,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
        
        # Create output structure similar to ESM2
        class ModelOutput:
            def __init__(self, logits, hidden_states):
                self.logits = logits
                self.hidden_states = hidden_states
        
        return ModelOutput(
            logits=outputs["logits"],
            hidden_states=(outputs.get("all_hiddens", None),) if "all_hiddens" in outputs else (outputs["last_hidden_state"],)
        )
