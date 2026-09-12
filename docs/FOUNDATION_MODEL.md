# Foundation model

ULTRON's foundation-model code is in backend/app/foundation. Its decoder, tokenizer, and trainer are ULTRON-owned. A model qualifies as ULTRON_FROM_SCRATCH only when its lineage declares RANDOM initialization, NONE pretrained weights, and NONE parent model.

The existing 112,960-parameter model is **proven** to perform a local random-init forward/train/checkpoint cycle. It is not proven to generalize, follow instructions, reason, code, or act as a useful assistant. It is M1 (loss-decrease) development infrastructure, not a quality milestone beyond that.

Qwen remains a THIRD_PARTY_REFERENCE_MODEL; its weights, tokenizer, outputs, and generated data are not inputs to this model lineage.
