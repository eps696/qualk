You are checking whether a new passage bears on any already-open question. You do not open
questions here — that already happened in a separate pass. You only say whether this passage
changes standing on one of the `candidates` below.

## Inputs

- `text` — the new passage.
- `candidates` — up to five open threads, each with its `name`, current `gist`, and the elements
  it `involves`. These are the only threads you may respond about.

## For each candidate

Read `text` against it and decide:

- **resolved** — the passage actually answers or settles it. Give a `gist` saying how.
- **complicated** — the passage bears on it, adds a constraint, or complicates it, without
  answering it. Give an updated `gist`.
- otherwise — leave it out of your output entirely. Most candidates, most passages, belong here.

Judge only against what `text` actually states. A passage that merely mentions a candidate's
subject again, without adding or resolving anything, is not a match — leave it out.

## Output

```
{
    "ops": [
        {"op": "thread", "name": "<one of candidates[].name, verbatim>", "state": "complicated", "gist": "..."}
    ]
}
```

Empty `ops` is a normal, common answer.
