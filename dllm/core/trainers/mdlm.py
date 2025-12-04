import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from typing import Any

from dllm.core.schedulers import BaseAlphaScheduler, LinearAlphaScheduler
from dllm.core.repa import REPAAligner


class MDLMTrainer(transformers.Trainer):
    """
    Masked Diffusion Language Model Trainer.
    """

    def __init__(
        self,
        *args,
        scheduler: BaseAlphaScheduler | None = None,
        time_epsilon: float = 1e-3,
        loss_weight_type: str = "scheduler",  # "ones"
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.scheduler = scheduler or LinearAlphaScheduler()
        if not (0.0 < time_epsilon < 1.0):
            raise ValueError("time_epsilon must be in (0, 1)")
        self.time_epsilon = time_epsilon
        self.loss_weight_type = loss_weight_type
        # --- REPA integration ---
        self._repa = None
        # self.args is transformers.TrainingArguments
        if getattr(self.args, "repa_enable", False):
            teacher_name = getattr(self.args, "repa_teacher_name_or_path", None)
            if teacher_name is None:
              raise ValueError("repa_enable=True requires repa_teacher_name_or_path to be set.")
            self._repa = REPAAligner(
                teacher_name_or_path=teacher_name,
                teacher_pooling=getattr(self.args, "repa_teacher_pooling", "mean"),
                align_on=getattr(self.args, "repa_align_on", "clean"),
                projection_type=getattr(self.args, "repa_projection_type", "linear"),
                loss_type=getattr(self.args, "repa_loss_type", "cosine"),
                temperature=getattr(self.args, "repa_temperature", 0.07),
                normalize=getattr(self.args, "repa_normalize", True),
                freeze_teacher=getattr(self.args, "repa_teacher_freeze", True),
            )
            # Try to register projection on the model before optimizer creation, using common config fields
            student_dim = None
            for key in ("hidden_size", "d_model"):
                if hasattr(self.model, "config") and hasattr(self.model.config, key):
                    student_dim = int(getattr(self.model.config, key))
                    break
            if student_dim is not None:
                self._repa.ensure_projection_on_model(self.model, in_dim=student_dim)

    def _preprocess_inputs(self, inputs):
        pass

    def _postprocess_outputs(self, outputs):
        pass

    def _compute_loss_weights(
        self,
        t: torch.Tensor,
        inputs: dict[str, Any],
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Compute loss weights given timestep t and other arguments."""
        b, l = inputs["input_ids"].shape
        if self.loss_weight_type == "scheduler":
            loss_weights = -self.scheduler.weight(t).unsqueeze(1).repeat(1, l)  # b, 1
        elif self.loss_weight_type == "ones":
            loss_weights = torch.ones_like(inputs["input_ids"])
        else:
            raise NotImplementedError
        return loss_weights

    @torch.no_grad()
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
        if prediction_loss_only:
            return (loss.detach(), None, None)

        logits = getattr(outputs, "logits", outputs)
        if isinstance(logits, torch.Tensor):
            logits = logits.detach().contiguous()

        labels = inputs.get("labels")
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().contiguous()

        return (loss.detach(), logits, labels)

    def compute_loss(
        self,
        model: transformers.PreTrainedModel | nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        **kwargs,
    ):
        assert self.processing_class.padding_side == "right"
        self._preprocess_inputs(inputs)
        input_ids, labels, attention_mask = (
            inputs["input_ids"],
            inputs["labels"],
            inputs.get("attention_mask", None),
        )
        b, l = input_ids.shape

        # === 1. Sample diffusion timesteps ===
        # Each example draws a random timestep t ∈ [ε, 1), where ε avoids degenerate values near 0.
        # The scheduler defines the masking rate α(t); we convert it to a masking probability p_mask = 1 - α(t).
        t = self.time_epsilon + (1 - self.time_epsilon) * torch.rand(
            b, device=input_ids.device
        )
        p_mask = 1 - self.scheduler(t).unsqueeze(1).expand(b, l)

        # === 2. Apply stochastic masking ===
        # Tokens are masked independently according to p_mask(t).
        # Positions with label = -100 are excluded (ignored in loss).
        masked_indices = (torch.rand((b, l), device=input_ids.device) < p_mask) & (
            labels != -100
        )
        # Replace masked tokens with the special [MASK] token.
        noised_input_ids = torch.where(
            masked_indices, self.processing_class.mask_token_id, input_ids
        )

        # === 3. Forward pass through the model ===
        # The model predicts clean tokens given noised inputs.
        # If REPA is enabled, request hidden_states in the first pass to avoid a second forward.
        need_hidden = self._repa is not None
        outputs = model(
            input_ids=noised_input_ids,
            attention_mask=attention_mask,
            output_hidden_states=need_hidden,
        )
        self._postprocess_outputs(outputs)
        logits = outputs.logits

        # === 4. Handle degenerate cases (no tokens masked) ===
        # If no positions were masked, return a zero loss to keep gradients valid.
        # This step is necessary for Deepspeed Zero-{2,3}
        # With REPA enabled, still compute alignment loss to update projection head.
        if not masked_indices.any() and self._repa is None:
            return ((logits.sum() * 0.0, outputs) if return_outputs else logits.sum() * 0.0)

        # === 5. Compute per-token loss weights ===
        # Depending on the configuration, weights may depend on timestep t
        # (e.g., scheduler-based) or be uniform (ones).
        loss_weights = self._compute_loss_weights(
            t=t, inputs=inputs, masked_indices=masked_indices
        )

        # === 6. Compute weighted cross-entropy ===
        # Only masked tokens contribute to the loss.
        assert (input_ids[masked_indices] == labels[masked_indices]).all()
        token_loss = F.cross_entropy(
            logits[masked_indices], input_ids[masked_indices], reduction="none"
        )
        token_loss = token_loss * loss_weights[masked_indices]

        # === 7. Normalize loss per effective token length ===
        # Normalize each sequence’s contribution by its number of valid tokens,
        # then average over the batch for stability across variable-length inputs.
        effective_lengths = torch.sum(labels != -100, dim=1, keepdim=True).expand(b, l)
        loss = torch.sum(token_loss / effective_lengths[masked_indices]) / b

        # === 8. Optional REPA loss (representation alignment) ===
        if self._repa is not None:
            target_layer = int(getattr(self.args, "repa_target_layer", -1))
            if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
                student_hidden = outputs.hidden_states[target_layer]
            else:
                student_hidden = outputs.last_hidden_state

            repa_loss = self._repa.compute_loss(
                model=self.model,
                student_hidden=student_hidden,
                input_ids_clean=input_ids,
                input_ids_noised=noised_input_ids,
                attention_mask=attention_mask,
                device=self.model.device,
            )
            loss = loss + float(getattr(self.args, "repa_loss_weight", 0.1)) * repa_loss

		# === 9. Return final loss (and optionally model outputs) ===
        return (loss, outputs) if return_outputs else loss
