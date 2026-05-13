"""
Atlas — Search-quality benchmark queries.

Five hand-picked queries with known expected matches, used to measure the
impact of search-pipeline changes. Each query specifies:

  - id          stable handle for tables and saved runs
  - endpoint    "search" or "ask" — which daemon endpoint to hit
  - params      query-string params for /search, JSON body for /ask
  - match       predicate used to find the expected result in the response

Match types:
  - path_exact         result.source_path == expanduser(value)
  - path_startswith    result.source_path starts with expanduser(value).
                       Used for "directory" expectations: any chunk under
                       that directory counts as a hit.
  - snippet_contains   case-insensitive substring match on result.snippet.
                       Used for iMessage-style "I want a chunk containing
                       this phrase" expectations.

Adding queries: append to QUERIES below. Keep ids unique. The runner
treats every entry as independent — no setup or teardown between them.
"""

QUERIES = [
    {
        "id": "q1-cs107-homework",
        "endpoint": "search",
        "params": {"q": "cs107 homework", "limit": 10},
        "match": {
            "type": "path_startswith",
            # The real on-disk path doubles Desktop (~/Desktop/DESKTOP/...).
            "value": "~/Desktop/DESKTOP/Stanford/1 Frosh/2 Frosh Winter/CS 107/cs107",
        },
    },
    {
        "id": "q2-ryan-rollins",
        "endpoint": "search",
        "params": {"q": "ryan rollins", "limit": 10, "source": "imessage"},
        "match": {
            "type": "snippet_contains",
            "value": "Ryan Rollins legacy game incoming",
        },
    },
    {
        "id": "q3-pearx",
        "endpoint": "search",
        "params": {"q": "pearx", "limit": 10},
        "match": {
            "type": "path_exact",
            "value": "~/Downloads/Stickies/PearX notes on investor meetings.txt",
        },
    },
    {
        "id": "q4-maxwell-demon",
        "endpoint": "search",
        "params": {"q": "Maxwell's demon", "limit": 10},
        "match": {
            "type": "path_exact",
            # File lives under assign1/, not directly under cs107/.
            "value": "~/Desktop/DESKTOP/Stanford/1 Frosh/2 Frosh Winter/CS 107/cs107/assign1/maxwell_demon.c",
        },
    },
    {
        "id": "q5-trainify-folder",
        "endpoint": "ask",
        "params": {"query": "find the folder with my trainify code", "limit": 10},
        "match": {
            "type": "path_startswith",
            "value": "~/Desktop/DESKTOP/Code/trainify-handover",
        },
    },
]
