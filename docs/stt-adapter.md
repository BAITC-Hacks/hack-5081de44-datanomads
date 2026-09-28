# STT integration status

`TASK-048` remains conditional and blocked on a real audio source. The
repository has no configured call audio stream, recording source, or transcript
API, so this change does not add an STT client, upload path, or synthetic audio
fixture.

The current operator flow accepts text already present on a ticket: Core reads
the ticket text for assist preview, and the Operator Panel keeps the original
text visible alongside the prediction and human decision controls. This P0
text flow remains the supported input path.

When a real source is available, the integration must pass its audio/recording
reference to an STT adapter, return a transcript to the existing ticket input,
and preserve human review. Until then, no transcript is inferred or attached to
a ticket, and the external STT dependency remains unmet.
