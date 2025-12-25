import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import EsmTokenizer, BatchEncoding
from typing import List, NamedTuple
from .amix_utils import load_amix_model


class ModelOutput(NamedTuple):
    """Output structure for AMix model compatible with ESM2."""
    logits: torch.Tensor
    hidden_states: tuple


class AMix(torch.nn.Module):
    def __init__(self, ckpt_path: str):
        """
        AMix model wrapper for directed evolution.
        
        Args:
            ckpt_path (str): Path to AMix checkpoint file.
            
        Raises:
            AssertionError: If ckpt_path is None
            FileNotFoundError: If checkpoint or config file doesn't exist
        """
        super(AMix, self).__init__()
        assert ckpt_path is not None, "AMix checkpoint path (ckpt_path) must be provided"
        
        # Load AMix model using utility function
        self.model = load_amix_model(ckpt_path, device='cpu')
        
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

    def forward(self, inputs: BatchEncoding) -> ModelOutput:
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
        return ModelOutput(
            logits=outputs["logits"],
            hidden_states=(outputs.get("all_hiddens", None),) if "all_hiddens" in outputs else (outputs["last_hidden_state"],)
        )

