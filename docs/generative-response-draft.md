# Generative response draft status

`TASK-049` remains conditional. `.env.example` keeps the optional LLM provider
disabled, and the Core/ML runtime has no response-generation client or API. The
placeholder provider settings do not constitute a working integration. No
generative draft endpoint or fabricated LLM response has been added.

P0 continues to use the existing approved response-template and manual-review
path. Assist preview reads an approved template for the selected language and
confirmed topic/service when available; otherwise it returns a manual-required
or unavailable state. Pulse does not send the response to the official CRM.

If an LLM provider is later configured for this feature, it must receive only
confirmed topic/service/priority, explicitly allow-listed facts, and approved
policy/template context. It must not receive ticket text or other PII without a
separate, justified and audited policy. Drafts must not add deadlines, official
status, causes, service phone numbers, or promises of resolution, and must stay
subject to operator review. The approved-template/manual path must remain usable
when the provider is disabled or unavailable.
