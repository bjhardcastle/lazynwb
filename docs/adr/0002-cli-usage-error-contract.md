# ADR 0002: CLI usage-error contract for agents

Status: Accepted

Date: 2026-05-04

Issue: #54

## Context

The first-party `lazynwb` CLI is an agent-facing replacement for the separate
MCP workflow described in #44. Its established contract is that command results
and machine-readable errors are written to stdout, debug logs are written to
stderr only when `--debug` is enabled, and automation can classify failures by
stable exit code.

Current usage errors already follow that shape. For example, running `lazynwb`
without a subcommand exits with code 2 and writes a JSON error object to stdout:
`error.code` is `usage_error`, `error.message` contains the argparse validation
message, and `error.details.usage` contains a usage string. Stderr is empty
unless `--debug` is present.

That is parseable, but it is not yet a good recovery contract for agents. The
usage string can be multi-line and long enough to be truncated or skimmed away
in agent contexts, and it does not provide a stable compact next action. Human
help output remains useful, but invalid invocations should still be primarily
machine-readable because they occur in scripts and agent shell tools.

## Decision

Keep structured JSON on stdout as the canonical response for invalid CLI
invocation. Do not switch usage errors to argparse's default stderr text, do not
emit full help text automatically, and do not rely on exit code alone.

For every invalid invocation handled as a usage error, an agent should receive:

- Exit code 2.
- A single newline-terminated JSON object on stdout.
- `error.code` set to `usage_error`.
- `error.message` set to the short parser validation message.
- `error.details.usage` preserving the current argparse usage string.
- Compact structured recovery fields in `error.details`, placed before the
  longer usage text when serialized with sorted keys.
- No stderr output unless `--debug` is enabled, in which case stderr may contain
  debug logs and stdout must remain parseable JSON.

The compact recovery fields should be stable and intentionally small. The first
implementation should add:

- `error.details.help_command`: the most specific help command known for the
  parser that failed, such as `lazynwb --help` or `lazynwb paths --help`.
- `error.details.recovery_hint`: one short sentence describing the next action.
- `error.details.valid_commands` for root-command failures, listing supported
  subcommands in their CLI order.

Subcommand option errors should point at that subcommand's help command rather
than dumping full help text. Root invocation errors, such as `lazynwb` with no
command or an unknown command, should include `valid_commands` so an agent can
choose the next command without parsing the usage line.

The existing stdout/stderr and exit-code contracts from #44 are therefore
preserved. The JSON error object is extended, not replaced.

## Rejected alternatives

Emit argparse-style human errors and help on stderr. This is familiar for
humans, but it would break the CLI's agent-first stdout/stderr split by putting
the actionable failure payload outside the machine-readable stream. It would
also make table or JSONL consumers handle a second error channel.

Automatically print full help text for invalid invocations. Help text is useful
when explicitly requested with `--help`, but it is long, mostly unstructured,
and likely to worsen the truncation problem that motivated #54.

Use exit code 2 with empty stdout. This is stable for shell scripts but gives
agents no recovery guidance and forces a second command just to discover the
valid command shape.

Replace the current JSON error with a shorter plain-text hint. That would make
the common failure easier to read, but it would abandon the machine-readable
error behavior required by #44.

Remove `error.details.usage` to reduce output size. That would be cleaner for
agents, but it could break any existing automation that already inspects the
usage field. Keeping `usage` while adding compact fields gives agents a shorter
stable path without removing current context.

Use documentation URLs as the primary recovery mechanism. Links can help
humans, but shell-native agents need offline, local, command-shaped guidance.
Docs may be added later, but `help_command` is the stable recovery pointer for
this slice.

## Consequences

Usage-error behavior remains consistent with the rest of the CLI error surface:
JSON on stdout, debug logs on stderr, and stable exit code 2. Agents gain a
small, predictable recovery object that is easier to consume than a multi-line
usage string.

The implementation should add focused CLI tests for root and subcommand usage
errors, including stdout JSON shape, empty stderr by default, debug logging on
stderr when requested, and preservation of `error.details.usage`. No runtime
change is made by this ADR itself.
