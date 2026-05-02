## Agent skills

### Issue tracker

Issues and PRDs are tracked in GitHub Issues for `bjhardcastle/lazynwb`. See `docs/agents/issue-tracker.md`.

### Triage labels

Use the default five-label triage vocabulary. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context repo: root `CONTEXT.md` and root `docs/adr/`. See `docs/agents/domain.md`.

## Coding
- performance is the highest priority
- prefer importing module namespaces over importing functions or classes directly, for example: `import lazynwb.utils as utils`, to maintain clarity about where functions and classes come from. The exception to this is certain stdlib imports, such as `typing` or `__future__`.
- add comprehensive debug logging.
- mark implementation details as private by prefixing with an underscore, for example: `_private_function()`. Agents should always check before modifying public interfaces or their tests.
