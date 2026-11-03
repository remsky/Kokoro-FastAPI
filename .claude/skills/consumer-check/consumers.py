"""Find downstream consumers of Kokoro-FastAPI on GitHub and search them for changed surface."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

SELF = "remsky/Kokoro-FastAPI"
DISCOVERY = ['"ghcr.io/remsky/kokoro-fastapi"', '"from api.src" kokoro']
WIDE = ['"remsky/Kokoro-FastAPI"', '8880 "audio/speech"']
MAX_QUERY_CHARS = 1000
PAUSE_S = 7
ROOT = Path(__file__).resolve().parents[3]


def code_search(query, max_pages=10):
    hits, page = [], 1
    while True:
        for _ in range(5):
            proc = subprocess.run(
                [
                    "gh",
                    "api",
                    "-X",
                    "GET",
                    "search/code",
                    "-H",
                    "Accept: application/vnd.github.text-match+json",
                    "-f",
                    f"q={query}",
                    "-f",
                    "per_page=100",
                    "-f",
                    f"page={page}",
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            if proc.returncode == 0:
                break
            if "rate limit" not in proc.stderr.lower():
                sys.exit(proc.stderr.strip())
            print("rate limited, waiting 60s", file=sys.stderr)
            time.sleep(60)
        else:
            sys.exit(f"still rate limited: {query}")
        time.sleep(PAUSE_S)
        items = json.loads(proc.stdout)["items"]
        hits += items
        if len(items) < 100 or page == max_pages:
            return hits
        page += 1


def list_consumers(wide, pages):
    repos = set()
    for query in DISCOVERY + (WIDE if wide else []):
        found = {i["repository"]["full_name"] for i in code_search(query, pages)}
        print(f"{query}: {len(found)} repos", file=sys.stderr)
        repos |= found
    repos.discard(SELF)
    return sorted(repos)


def own_files():
    proc = subprocess.run(
        ["git", "ls-files", "api", "web"], cwd=ROOT, capture_output=True, text=True
    )
    return set(proc.stdout.split())


def is_copy(path, own):
    parts = path.split("/")
    return any("/".join(parts[i:]) in own for i in range(len(parts)))


def pack_queries(term, repos):
    base = f'"{term}"'
    query = base
    for repo in repos:
        qualifier = f" repo:{repo}"
        if len(query) + len(qualifier) > MAX_QUERY_CHARS and query != base:
            yield query
            query = base
        query += qualifier
    if query != base:
        yield query


def search_consumers(terms, repos):
    own = own_files()
    copies = set()
    for term in terms:
        hit_repos = set()
        for query in pack_queries(term, repos):
            for item in code_search(query):
                repo = item["repository"]["full_name"]
                if is_copy(item["path"], own):
                    copies.add(repo)
                    continue
                hit_repos.add(repo)
                lines = [
                    ln.strip()
                    for match in item.get("text_matches") or []
                    for ln in match["fragment"].splitlines()
                ]
                line = next((ln for ln in lines if term.lower() in ln.lower()), "")
                print(f"{term}\t{repo}\t{item['html_url']}\t{line[:160]}")
        print(f"{term}: {len(hit_repos)} of {len(repos)} consumers", file=sys.stderr)
    print(
        f"skipped hits in {len(copies)} repos that copy Kokoro's own api/ or web/ files",
        file=sys.stderr,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    listing = sub.add_parser("list", help="print consumer repos, one per line")
    listing.add_argument(
        "--full",
        action="store_true",
        help="read all 10 result pages per query instead of the best 3 (slower, finds more)",
    )
    listing.add_argument(
        "--wide",
        action="store_true",
        help="also match clone URLs and port 8880 clients (slow, noisier)",
    )
    searching = sub.add_parser("search", help="search consumer repos for each term")
    searching.add_argument("terms", nargs="+")
    searching.add_argument("--repos", required=True, help="file from the list command")
    args = parser.parse_args()

    if args.command == "list":
        print("\n".join(list_consumers(args.wide, 10 if args.full else 3)))
    else:
        with open(args.repos, encoding="utf-8") as f:
            repos = [ln.strip() for ln in f if ln.strip()]
        search_consumers(args.terms, repos)


if __name__ == "__main__":
    main()
