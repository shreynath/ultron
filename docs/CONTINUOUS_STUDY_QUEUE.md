# Continuous Study Queue

The Level 1 Study Queue is the only self-directed acquisition route. It is a
durable ordered SQLite queue under `ULTRON_HOME/learning/study_queue.sqlite3`,
not a conversational promise and not parameter training.

Each queued target defines a baseline question, a paraphrase, a deterministic
answer key, candidate information, and an HTTPS external source. The controller
records the baseline before acquisition; fetches, content-addresses, and stores
the external document; requires the answer key to be supported by that document;
saves a semantic memory with source provenance; discards the acquisition context; then
uses a new memory-store instance for exact and paraphrased retrieval tests.

The answer evaluator is `DETERMINISTIC_NORMALIZED_SUBSTRING_V1`. Generated
output is never accepted as either source material or ground truth. A target is
`LEARNING_DEMONSTRATED` only when the baseline is incorrect and both independent
retests are correct. Source failure is `UNKNOWN`; a valid source with a failed
retest is `LEARNING_FAILED`.

The queue starts in supervised mode. An unattended run is refused until an
operator records a human-graded calibration of at least ten items and at least
90% evaluator agreement. This guard deliberately prevents an unvalidated judge
from accumulating unsupported learning claims. Queue targets, calibration, and
events are available from `/api/local-intelligence/study-queue`.

The first two supervised production events are
`8b8da863-5f30-4cae-9281-5105260f735e` and
`474de8a4-db22-4161-b40e-c8b5e277d5fe`. The official RFC 1035 document was
hashed as
`d14ae809fc9b41bbcae26bb38c937c9515808b944f3252b00de9fe0e95f4fdfb` and
demonstrated persistent retrieval of the DNS message identifier width and DNS
zone-transfer transport after fresh contexts. This supports a narrow Level 1
memory-learning claim only; it does not demonstrate behavioral or parameter
learning.
