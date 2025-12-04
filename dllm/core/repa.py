import torch
import torch.nn as nn
from typing import Optional, Literal, Tuple

import transformers


PoolingStrategy = Literal["mean", "cls", "last_token"]
ProjectionType = Literal["linear", "mlp"]
LossType = Literal["cosine", "mse"]
AlignOn = Literal["clean", "noised"]


class REPATeacherEncoder(nn.Module):
	"""
	Wraps a frozen external encoder (e.g., BERT/LLaMA/Qwen) and provides pooled representations.
	"""

	def __init__(
		self,
		name_or_path: str,
		pooling: PoolingStrategy = "mean",
		freeze: bool = True,
		normalize: bool = True,
	):
		super().__init__()
		self.pooling = pooling
		self.normalize = normalize

		# Try base encoder; fallback to causal LM if needed
		try:
			self.encoder = transformers.AutoModel.from_pretrained(name_or_path)
		except Exception:
			self.encoder = transformers.AutoModelForCausalLM.from_pretrained(name_or_path)

		if freeze:
			for p in self.encoder.parameters():
				p.requires_grad = False
			self.encoder.eval()

	@property
	def hidden_size(self) -> int:
		return int(getattr(self.encoder.config, "hidden_size"))

	@torch.no_grad()
	def forward(
		self,
		input_ids: torch.LongTensor,
		attention_mask: Optional[torch.Tensor] = None,
	) -> torch.Tensor:
		"""
		Returns pooled features shape [B, D].
		"""
		out = self.encoder(
			input_ids=input_ids,
			attention_mask=attention_mask,
			output_hidden_states=True,
		)
		hidden = out.last_hidden_state  # [B,L,D]

		if self.pooling == "cls":
			# If CLS not present (e.g., LLaMA), fall back to first token
			pooled = hidden[:, 0]
		elif self.pooling == "last_token":
			if attention_mask is not None:
				lengths = attention_mask.long().sum(-1) - 1
				pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), lengths]
			else:
				pooled = hidden[:, -1]
		else:
			# mean pool over valid tokens
			if attention_mask is None:
				pooled = hidden.mean(dim=1)
			else:
				mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
				sum_hidden = (hidden * mask).sum(dim=1)
				denom = mask.sum(dim=1).clamp_min(1.0)
				pooled = sum_hidden / denom

		if self.normalize:
			pooled = nn.functional.normalize(pooled, p=2, dim=-1)
		return pooled


class REPAProjection(nn.Module):
	"""
	Projection head mapping model hidden size -> teacher hidden size.
	"""

	def __init__(
		self,
		in_features: int,
		out_features: int,
		projection_type: ProjectionType = "linear",
	):
		super().__init__()
		if projection_type == "linear":
			self.net = nn.Linear(in_features, out_features, bias=True)
		elif projection_type == "mlp":
			mid = max(min(4 * min(in_features, out_features), 2048), 64)
			self.net = nn.Sequential(
				nn.Linear(in_features, mid, bias=True),
				nn.GELU(),
				nn.Linear(mid, out_features, bias=True),
			)
		else:
			raise ValueError(f"Unknown projection_type={projection_type}")

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.net(x)


class REPAAligner:
	"""
	Manages teacher encoding, projection head on the student model, and computes REPA loss.
	"""

	def __init__(
		self,
		teacher_name_or_path: str,
		teacher_pooling: PoolingStrategy = "mean",
		align_on: AlignOn = "clean",
		projection_type: ProjectionType = "linear",
		loss_type: LossType = "cosine",
		temperature: float = 0.07,
		normalize: bool = True,
		freeze_teacher: bool = True,
		teacher_device: str = "auto",
		teacher_dtype: Optional[str] = None,
	):
		self.teacher = REPATeacherEncoder(
			name_or_path=teacher_name_or_path,
			pooling=teacher_pooling,
			freeze=freeze_teacher,
			normalize=normalize,
		)
		self.align_on = align_on
		self.projection_type = projection_type
		self.loss_type = loss_type
		self.temperature = float(temperature)
		self.normalize = normalize
		self._projection_ref: Optional[REPAProjection] = None
		self.teacher_device_pref = teacher_device
		self.teacher_dtype_pref = teacher_dtype
		self._teacher_placed = False

	def _resolve_dtype(self, name: Optional[str]) -> Optional[torch.dtype]:
		if name is None:
			return None
		name = name.lower()
		if name in ("bf16", "bfloat16"):
			return torch.bfloat16
		if name in ("fp16", "float16", "half"):
			return torch.float16
		if name in ("fp32", "float32", "f32"):
			return torch.float32
		return None

	def _ensure_teacher_placement(self, student_device: torch.device):
		if self._teacher_placed:
			return
		pref = (self.teacher_device_pref or "auto").lower()
		if pref == "student":
			target_device = student_device
		elif pref == "auto":
			target_device = student_device if student_device.type == "cuda" else torch.device("cpu")
		elif pref == "cuda":
			target_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		else:
			# allow "cuda:1", "cpu", etc.
			try:
				target_device = torch.device(pref)
			except Exception:
				target_device = student_device
		dtype = self._resolve_dtype(self.teacher_dtype_pref)
		# Move teacher once
		self.teacher.to(device=target_device)  # type: ignore[operator]
		if dtype is not None:
			self.teacher.to(dtype=dtype)  # type: ignore[operator]
		self._teacher_placed = True

	def ensure_projection_on_model(self, model: nn.Module, in_dim: int) -> nn.Module:
		"""
		Creates and attaches the projection head to the student model (as `repa_head`)
		if it doesn't already exist. Returns the projection module.
		"""
		if hasattr(model, "repa_head") and isinstance(model.repa_head, nn.Module):
			self._projection_ref = model.repa_head  # type: ignore[attr-defined]
			return model.repa_head  # type: ignore[attr-defined]
		proj = REPAProjection(
			in_features=int(in_dim),
			out_features=self.teacher.hidden_size,
			projection_type=self.projection_type,
		)
		# Register on model so parameters are optimized and saved with checkpoints
		setattr(model, "repa_head", proj)
		self._projection_ref = proj
		return proj

	def _pool_student_tokens(
		self,
		hidden: torch.Tensor,  # [B,L,D]
		attention_mask: Optional[torch.Tensor],
	) -> torch.Tensor:
		# Match teacher pooling "mean" by default for robustness across tokenizers
		if attention_mask is None:
			return hidden.mean(dim=1)
		mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
		sum_hidden = (hidden * mask).sum(dim=1)
		denom = mask.sum(dim=1).clamp_min(1.0)
		return sum_hidden / denom

	def compute_loss(
		self,
		model: nn.Module,
		student_hidden: torch.Tensor,  # [B,L,D]
		input_ids_clean: torch.LongTensor,
		input_ids_noised: torch.LongTensor,
		attention_mask: Optional[torch.Tensor] = None,
		device: Optional[torch.device] = None,
	) -> torch.Tensor:
		"""
		Computes alignment loss between student pooled representation (projected) and teacher pooled features.
		"""
		student_device = student_hidden.device
		self._ensure_teacher_placement(student_device)

		with torch.no_grad():
			teacher_ids = input_ids_clean if self.align_on == "clean" else input_ids_noised
			# Move inputs to the teacher's device to avoid CPU/CUDA mismatch
			try:
				teacher_device = next(self.teacher.parameters()).device  # type: ignore[assignment]
			except StopIteration:
				teacher_device = student_device
			teacher_ids = teacher_ids.to(teacher_device)
			teacher_mask = attention_mask.to(teacher_device) if attention_mask is not None else None
			# Guard against tokenizer/vocab mismatch. If ids exceed teacher vocab, skip REPA (zero-loss).
			teacher_vocab = int(getattr(self.teacher.encoder.config, "vocab_size", -1))
			if teacher_vocab > 0:
				max_id = int(teacher_ids.max().item())
				min_id = int(teacher_ids.min().item())
				if min_id < 0 or max_id >= teacher_vocab:
					return student_hidden.sum() * 0.0  # keep graph dependency on student
			teacher_feat = self.teacher(teacher_ids, teacher_mask)  # [B,Dt]
			# Bring teacher features back to student's device for loss computation
			if teacher_feat.device != student_device:
				teacher_feat = teacher_feat.to(student_device)

		student_pooled = self._pool_student_tokens(student_hidden, attention_mask)  # [B,Ds]
		proj_head = self.ensure_projection_on_model(model, in_dim=student_pooled.size(-1))
		student_proj = proj_head(student_pooled)  # [B,Dt]

		if self.normalize:
			student_proj = nn.functional.normalize(student_proj, p=2, dim=-1)

		if self.loss_type == "cosine":
			# Loss = 1 - cosine_similarity
			loss = 1.0 - torch.nn.functional.cosine_similarity(student_proj, teacher_feat, dim=-1)
			return loss.mean() # type: ignore[return-value]
		elif self.loss_type == "mse":
			return torch.nn.functional.mse_loss(student_proj, teacher_feat)
		else:
			raise ValueError(f"Unknown loss_type={self.loss_type}")


