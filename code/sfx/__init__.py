"""SFX package. Importing it routes the NeMo Transformer encoder through its dense-mask SDPA path:
the compiled FlexAttention kernel needs more shared memory than consumer Blackwell GPUs (RTX 5090) have.
The SDPA path computes the same masked attention (it is NeMo's export path)."""
from nemo.collections.asr.modules import transformer_encoder as _te

_orig_forward_internal = _te.TransformerEncoder.forward_internal


def _forward_internal_sdpa(self, audio_signal, length, bypass_pre_encode=False, for_export=False):
    return _orig_forward_internal(self, audio_signal, length, bypass_pre_encode=bypass_pre_encode, for_export=True)


_te.TransformerEncoder.forward_internal = _forward_internal_sdpa

# allow config-driven instantiation of our own classes (nn.Module / ModelPT subclasses only, as NeMo checks)
from nemo.core.classes import common as _common

if "sfx." not in _common.ALLOWED_TARGET_PREFIXES:
    _common.ALLOWED_TARGET_PREFIXES.append("sfx.")
