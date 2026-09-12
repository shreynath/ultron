# Cognitive execution

`app.brain.cognitive_execution` persists a bounded task object and validates model outputs as typed assessment, plan, tool request, observation interpretation, verification, reflection, and final response records. Tool calls require an allowlisted task tool and a local executor. Limit failures are durable; malformed outputs fail safely.

This complements the existing executive/coordinator and evidence layers. External material remains provenance-marked data, not instruction.
