---
name: consumer-check
description: "Checking whether a Kokoro-FastAPI change breaks downstream users (image consumers, HTTP clients, in-process shims). Use before merging a behavior change that can't be compared against the OpenAI, ElevenLabs, or Cartesia client libraries, or when asked who uses X feature."
---

Read [AGENTS.md](../../../AGENTS.md) and [CONTRIBUTING.md](../../../CONTRIBUTING.md) first, house rules live there.

# Consumer check

A consumer is a public GitHub repo that runs or calls Kokoro-FastAPI. A search of this repo only shows that Kokoro itself no longer uses something. Consumers often use the parts that look unused from inside: test and health endpoints, `api.src` internals, status codes, and empty responses.

Skip it if the change can be compared against the OpenAI, ElevenLabs, or Cartesia client libraries.

A GitHub search for the changed endpoint or field alone finds almost nothing. First find the consumers by how they get Kokoro. Then search those repos.

## Steps

1. Make a list of the changed surface from the diff. This includes endpoints, request fields, status codes, response headers, response body shape, env vars, `api.src` import paths, function signatures, and file paths in the image.
2. Log in to the GitHub CLI with `gh auth login`.
3. Make the consumer list. By default the script reads the best 3 pages of results for each query, which takes about a minute.
   ```bash
   uv run python .claude/skills/consumer-check/consumers.py list > /tmp/consumers.txt
   ```
   A full search is a good idea before a release, or when the change rejects requests that worked before or removes something. It is not required. Add `--full` to read all 10 pages. In November 2026 the full search found about 580 consumer repos.
   If the change removes something, you can also add `--wide`. It also finds clone URLs and clients on port 8880, but it is slower and gives more false hits.
   If you have a list from an earlier check, you can use it again and skip this step.
4. Search the consumers for each changed item and for each coupling probe below. The script prints one line per hit: term, repo, file link, matching line. GitHub code search allows about 10 requests a minute, so each term takes 1 to 3 minutes. Hits in copies of Kokoro's own `api/` or `web/` files are skipped.
   ```bash
   uv run python .claude/skills/consumer-check/consumers.py search download_format audio/speech v1/test 8880/health api.src kokoro-fastapi --repos /tmp/consumers.txt
   ```
   Do not start a term with `/`. Git Bash on Windows changes it into a file path.
5. Read each hit file with `gh api repos/OWNER/REPO/contents/PATH`. Then give each consumer a verdict.

## Coupling probes

Coupling is how a consumer depends on Kokoro. Search these terms on every check, because they find dependencies that the diff does not show.

| Term | Finds |
|---|---|
| `audio/speech` | HTTP client code. Read how it handles a non-2xx status, an empty body, and response headers. A client that stops on any error breaks when a 200 becomes a 400 |
| `v1/test`, `8880/health` | Healthchecks and backend detection |
| `api.src` | Shims, which are code that runs inside the image and imports Kokoro internals. For these, import paths and signatures (including argument order) are public |
| `kokoro-fastapi` | The image tag. A `latest` tag gets the change at the next release. A pinned tag gets it when the consumer updates the pin |

## Report

For each consumer, give the coupling type, the file link, the changed items it uses, and a verdict: SAFE, AFFECTED or UNCLEAR. If a repo has no hits, write "nothing found". Do not leave it out.

If a consumer is AFFECTED, or it has a risky pattern such as a client that stops on a 400, write it in the PR description.

## Do not

- Do not say "no consumers" after a search for the changed name only.
- Do not open issues or write comments on consumer repos.
- Do not clone consumer repos into this workspace.
