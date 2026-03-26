#!/usr/bin/env python3
"""
AmericanStories Hedonometer Pipeline
=====================================
Reads locally downloaded AmericanStories JSON files, groups articles by
newspaper (LCCN), applies hedonometer sentiment scoring, geocodes each
newspaper via Nominatim, and outputs a self-contained interactive HTML map.

Expected directory layout (relative to this script):
  data/
    faro_1800/
      1800-01-...json
      ...
    faro_1801/
      ...
    ...
  Hedonometer.csv

Run this script on a machine with:
  - pip install requests tqdm

Usage:
  python process_americanstories.py

Output:
  newspaper_happiness_map.html  — self-contained HTML file with embedded data
"""

from __future__ import annotations
import os, sys, re, json, random, time
from collections import defaultdict

# ── Dependency checks ────────────────────────────────────────────────────────
try:
    import requests
except ImportError:
    sys.exit("Missing: pip install requests")
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kw: x   # graceful fallback

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
START_YEAR  = 1800  # overridden by --start / --end args
END_YEAR    = 1850
SAMPLE_SIZE       = 10000  # max articles per (newspaper, year)
MIN_SCORED_WORDS  = 1000   # min labMT-matched words required to include a newspaper
OUTPUT_HTML      = "newspaper_happiness_map.html"
GEO_CACHE_FILE   = "geo_cache.json"
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Directory containing faro_YYYY/ sub-folders, relative to this script
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Local Hedonometer CSV — must have columns "Word" and "Happiness Score"
HEDONOMETER_CSV = os.path.join(os.path.dirname(__file__), "Hedonometer.csv")

# Optional exceptions file — lines of the form:  Newspaper title –– city, ST
EXCEPTIONS_FILE = os.path.join(os.path.dirname(__file__), "exceptions.txt")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Load labMT happiness word list from local CSV
# ─────────────────────────────────────────────────────────────────────────────

def load_labmt(csv_path: str = HEDONOMETER_CSV) -> dict[str, float]:
    """
    Load labMT happiness scores from the local Hedonometer CSV.
    Returns dict mapping word (lowercase) -> happiness score (1–9 scale).
    """
    import csv as _csv

    if not os.path.isfile(csv_path):
        sys.exit(f"✗ Hedonometer CSV not found: {csv_path!r}")

    scores = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for row in _csv.DictReader(f):
            word  = row.get("Word", "").strip().lower()
            score = row.get("Happiness Score", "").strip()
            if word and score:
                try:
                    scores[word] = float(score)
                except ValueError:
                    pass

    if not scores:
        sys.exit(f"✗ No scores parsed from {csv_path!r} — check column names.")

    print(f"  ✓ {len(scores):,} words loaded from {csv_path}")
    return scores


def hedonometer_score(text: str, labmt: dict) -> tuple[float, int] | tuple[None, int]:
    """
    Average labMT happiness score for all scoreable words in text.
    Returns (score, n_scored_words) where score is on the 1–9 scale,
    or (None, 0) if no labMT words were found.
    """
    words  = re.findall(r"[a-z']+", text.lower())
    scores = [labmt[w] for w in words if w in labmt]
    if scores:
        return sum(scores) / len(scores), len(scores)
    return None, 0


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Geocode newspapers
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2a: Load manual exceptions for newspapers that couldn't be geocoded
# ─────────────────────────────────────────────────────────────────────────────

def load_exceptions(path: str = EXCEPTIONS_FILE) -> dict[str, tuple[str, str]]:
    """
    Parse the exceptions file into a lookup dict.

    Expected format (one per line):
        Newspaper title –– city, ST

    Returns dict mapping normalised title -> (city, state_full_name).
    If the file doesn't exist, returns an empty dict silently.
    """
    lookup = {}
    if not os.path.isfile(path):
        return lookup

    state_abbrev = {
        "AL": "Alabama", "AK": "Arkansas",  # note: file uses AK for Arkansas
        "AR": "Arkansas", "AZ": "Arizona",
        "CA": "California", "CO": "Colorado", "CT": "Connecticut",
        "DC": "District of Columbia", "DE": "Delaware", "FL": "Florida",
        "GA": "Georgia", "HI": "Hawaii", "IA": "Iowa", "ID": "Idaho",
        "IL": "Illinois", "IN": "Indiana", "KS": "Kansas", "KY": "Kentucky",
        "LA": "Louisiana", "MA": "Massachusetts", "MD": "Maryland",
        "ME": "Maine", "MI": "Michigan", "MN": "Minnesota", "MO": "Missouri",
        "MS": "Mississippi", "MT": "Montana", "NC": "North Carolina",
        "ND": "North Dakota", "NE": "Nebraska", "NH": "New Hampshire",
        "NJ": "New Jersey", "NM": "New Mexico", "NV": "Nevada",
        "NY": "New York", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon",
        "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
        "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah",
        "VA": "Virginia", "VT": "Vermont", "WA": "Washington",
        "WI": "Wisconsin", "WV": "West Virginia", "WY": "Wyoming",
    }

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "––" not in line:
                continue
            title_part, location_part = line.split("––", 1)
            title_norm = _normalise(title_part.strip())
            location_part = location_part.strip()
            if "," in location_part:
                city, state_code = [p.strip() for p in location_part.rsplit(",", 1)]
                state = state_abbrev.get(state_code.upper(), state_code)
            else:
                city  = location_part
                state = ""
            lookup[title_norm] = (city, state)

    print(f"  ✓ {len(lookup)} exception entries loaded from {path}")
    return lookup


def _normalise(text: str) -> str:
    """Lowercase and strip punctuation for fuzzy title matching."""
    return re.sub(r"[^a-z0-9\s]", "", text.lower()).strip()


# Module-level exceptions lookup (populated in main)
_exceptions: dict[str, tuple[str, str]] = {}


def lookup_exception(title: str) -> tuple[str, str] | tuple[None, None]:
    """
    Check the exceptions dict for a newspaper title.
    Uses substring matching on normalised strings to tolerate
    differences between the exception file and lccn.title formats.
    """
    if not _exceptions:
        return None, None
    norm = _normalise(title)
    # Exact normalised match first
    if norm in _exceptions:
        return _exceptions[norm]
    # Substring match: exception key appears in the lccn title or vice-versa
    for key, val in _exceptions.items():
        if key in norm or norm in key:
            return val
    return None, None


# State full-name → (lat, lon) centroid — used when Nominatim can't find the city
STATE_CENTROIDS = {
    "Alabama": (32.806671, -86.791130), "Alaska": (61.370716, -152.404419),
    "Arizona": (33.729759, -111.431221), "Arkansas": (34.969704, -92.373123),
    "California": (36.116203, -119.681564), "Colorado": (39.059811, -105.311104),
    "Connecticut": (41.597782, -72.755371), "Delaware": (39.318523, -75.507141),
    "Florida": (27.766279, -81.686783), "Georgia": (33.040619, -83.643074),
    "Idaho": (44.240459, -114.478828), "Illinois": (40.349457, -88.986137),
    "Indiana": (39.849426, -86.258278), "Iowa": (42.011539, -93.210526),
    "Kansas": (38.526600, -96.726486), "Kentucky": (37.668140, -84.670067),
    "Louisiana": (31.169960, -91.867805), "Maine": (44.693947, -69.381927),
    "Maryland": (39.063946, -76.802101), "Massachusetts": (42.230171, -71.530106),
    "Michigan": (43.326618, -84.536095), "Minnesota": (45.694454, -93.900192),
    "Mississippi": (32.741646, -89.678696), "Missouri": (38.456085, -92.288368),
    "Montana": (46.921925, -110.454353), "Nebraska": (41.125370, -98.268082),
    "Nevada": (38.313515, -117.055374), "New Hampshire": (43.452492, -71.563896),
    "New Jersey": (40.298904, -74.521011), "New Mexico": (34.840515, -106.248482),
    "New York": (42.165726, -74.948051), "North Carolina": (35.630066, -79.806419),
    "North Dakota": (47.528912, -99.784012), "Ohio": (40.388783, -82.764915),
    "Oklahoma": (35.565342, -96.928917), "Oregon": (44.572021, -122.070938),
    "Pennsylvania": (40.590752, -77.209755), "Rhode Island": (41.680893, -71.511780),
    "South Carolina": (33.856892, -80.945007), "South Dakota": (44.299782, -99.438828),
    "Tennessee": (35.747845, -86.692345), "Texas": (31.054487, -97.563461),
    "Utah": (40.150032, -111.862434), "Vermont": (44.045876, -72.710686),
    "Virginia": (37.769337, -78.169968), "Washington": (47.400902, -121.490494),
    "West Virginia": (38.491226, -80.954453), "Wisconsin": (44.268543, -89.616508),
    "Wyoming": (42.755966, -107.302490), "District of Columbia": (38.897438, -77.026817),
}

# Cache: "city|state" -> (lat, lon)
_geo_cache: dict[str, tuple] = {}


def geocode_city_state(city: str, state: str) -> tuple[float, float] | tuple[None, None]:
    """
    Geocode a city/state pair via Nominatim, falling back to the state centroid.
    Rate-limited to 1 req/sec to respect Nominatim ToS.
    """
    if not city and not state:
        return None, None

    cache_key = f"{city}|{state}"
    if cache_key in _geo_cache:
        return _geo_cache[cache_key]

    if city:
        query = f"{city}, {state}, USA" if state else f"{city}, USA"
        try:
            resp = requests.get(
                "https://nominatim.openstreetmap.org/search",
                params={"q": query, "format": "json", "limit": 1, "countrycodes": "us"},
                headers={"User-Agent": "AmericanStoriesHedonometer/1.0"},
                timeout=10,
            )
            resp.raise_for_status()
            results = resp.json()
            if results:
                coords = (float(results[0]["lat"]), float(results[0]["lon"]))
                _geo_cache[cache_key] = coords
                time.sleep(1)
                return coords
        except Exception:
            pass

    # Fall back to state centroid
    if state in STATE_CENTROIDS:
        _geo_cache[cache_key] = STATE_CENTROIDS[state]
        return STATE_CENTROIDS[state]

    _geo_cache[cache_key] = (None, None)
    return None, None


def extract_city_from_title(title: str) -> str:
    """
    Pull the city name out of an lccn title string such as:
      'New-York daily tribune. [volume] (New-York [N.Y.]) 1842-1866'
      'Charleston mercury. (Charleston, S.C.) 1822-1868'
    Returns the city string, or "" if not found.
    """
    # Pattern A: (City, State.)
    m = re.search(r"\(([A-Za-z][A-Za-z\s\-\.]+?),\s*[A-Z]", title)
    if m:
        return m.group(1).strip()
    # Pattern B: (City [ST.])
    m = re.search(r"\(([A-Za-z][A-Za-z\s\-]+?)\s*\[", title)
    if m:
        return m.group(1).strip()
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Load and process local JSON files year by year
# ─────────────────────────────────────────────────────────────────────────────

def process_year(year: int, labmt: dict) -> list[dict]:
    """
    Read all JSON files in data/faro_{year}/, group article text by LCCN,
    reservoir-sample up to SAMPLE_SIZE articles per newspaper, compute
    hedonometer scores, and geocode.
    Returns list of records: {newspaper, city, state, lat, lon, year, score, n_articles}
    """
    print(f"\n── Year {year} ──────────────────────────")

    year_dir = os.path.join(DATA_DIR, f"faro_{year}")
    if not os.path.isdir(year_dir):
        print(f"  ✗ Directory not found: {year_dir}")
        return []

    json_files = [f for f in os.listdir(year_dir) if f.endswith(".json")]
    if not json_files:
        print(f"  ✗ No JSON files in {year_dir}")
        return []

    # buckets keyed by lccn identifier string
    # each bucket: {title, state, texts[], count}
    buckets: dict[str, dict] = {}

    for fname in tqdm(json_files, desc=f"  reading {year}", unit=" files"):
        fpath = os.path.join(year_dir, fname)
        try:
            with open(fpath, encoding="utf-8") as f:
                doc = json.load(f)
        except Exception:
            continue

        lccn_meta = doc.get("lccn", {})
        lccn_id   = lccn_meta.get("lccn", "").strip()
        title     = lccn_meta.get("title", "").strip()
        state     = lccn_meta.get("state", "").strip()

        if not lccn_id:
            continue

        # Collect article text from all bboxes classified as "article"
        article_texts = [
            bbox["raw_text"]
            for bbox in doc.get("bboxes", [])
            if bbox.get("class") == "article" and bbox.get("raw_text", "").strip()
        ]

        if not article_texts:
            continue

        if lccn_id not in buckets:
            buckets[lccn_id] = {
                "title": title,
                "state": state,
                "texts": [],
                "count": 0,
            }

        bucket = buckets[lccn_id]
        for text in article_texts:
            bucket["count"] += 1
            # Reservoir sampling across all articles seen so far
            if len(bucket["texts"]) < SAMPLE_SIZE:
                bucket["texts"].append(text)
            else:
                j = random.randint(0, bucket["count"] - 1)
                if j < SAMPLE_SIZE:
                    bucket["texts"][j] = text

    print(f"  → {len(buckets):,} newspapers found")

    # Score and geocode each newspaper
    results = []
    for lccn_id, bucket in buckets.items():
        combined = " ".join(bucket["texts"])
        score, n_scored = hedonometer_score(combined, labmt)
        if score is None:
            print(f"    skip (no scoreable words): {lccn_id}")
            continue
        if n_scored < MIN_SCORED_WORDS:
            print(f"    skip (only {n_scored} scored words < floor {MIN_SCORED_WORDS}): {lccn_id}")
            continue

        title = bucket["title"]
        state = bucket["state"]
        city  = extract_city_from_title(title)
        lat, lon = geocode_city_state(city, state)

        # Fall back to the manually curated exceptions file
        if lat is None or lon is None:
            exc_city, exc_state = lookup_exception(title)
            if exc_city or exc_state:
                lat, lon = geocode_city_state(
                    exc_city  or city,
                    exc_state or state,
                )
                if lat is not None:
                    city  = exc_city  or city
                    state = exc_state or state

        if lat is None or lon is None:
            print(f"    skip (geocode failed) city={city!r} state={state!r} — {title}")
            continue

        results.append({
            "newspaper":      title,
            "city":           city,
            "state":          state,
            "lat":            round(lat, 4),
            "lon":            round(lon, 4),
            "year":           year,
            "score":          round(score, 4),
            "n_articles":     bucket["count"],
            "n_sampled":      len(bucket["texts"]),
            "n_scored_words": n_scored,
        })

    print(f"  → {len(results):,} newspapers geocoded & scored")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main(existing_records: list | None = None):
    print("=" * 60)
    print("AmericanStories Hedonometer Pipeline")
    print(f"  Years:       {START_YEAR} – {END_YEAR}")
    print(f"  Sample size: {SAMPLE_SIZE} articles per newspaper/year")
    print(f"  Data dir:    {DATA_DIR}")
    print("=" * 60)

    # 1. Load word list from local CSV
    labmt = load_labmt()

    # 2. Load geocoding exceptions
    global _exceptions
    _exceptions = load_exceptions()

    # 3. Load persistent geocode cache from disk (if it exists)
    global _geo_cache
    if os.path.exists(GEO_CACHE_FILE):
        with open(GEO_CACHE_FILE) as f:
            raw = json.load(f)
        # JSON serialises tuples as lists; convert back to tuples
        _geo_cache = {k: tuple(v) for k, v in raw.items()}
        print(f"✓ Loaded {len(_geo_cache):,} cached geocode entries from {GEO_CACHE_FILE}")
    else:
        print(f"  (No geocode cache found at {GEO_CACHE_FILE} – will build from scratch)")

    # 4. Determine which years still need processing
    already_done: set[int] = set()
    if existing_records:
        already_done = {r["year"] for r in existing_records}
        years_to_skip = sorted(already_done & set(range(START_YEAR, END_YEAR + 1)))
        if years_to_skip:
            print(f"  Skipping {len(years_to_skip)} already-present year(s): "
                  f"{years_to_skip[0]}–{years_to_skip[-1]}")

    years_to_run = [y for y in range(START_YEAR, END_YEAR + 1) if y not in already_done]
    if not years_to_run:
        print("  Nothing new to process – all years already present in existing data.")
        all_records = existing_records
    else:
        print(f"  Processing {len(years_to_run)} new year(s): {years_to_run[0]}–{years_to_run[-1]}")
        new_records = []
        for year in years_to_run:
            new_records.extend(process_year(year, labmt))
        all_records = (existing_records or []) + new_records

    # 5. Persist geocode cache so Nominatim is never called twice for the same location
    with open(GEO_CACHE_FILE, "w") as f:
        json.dump({k: list(v) for k, v in _geo_cache.items()}, f, indent=2)
    print(f"✓ Saved {len(_geo_cache):,} geocode entries to {GEO_CACHE_FILE}")

    if not all_records:
        sys.exit("No records were produced. Check that DATA_DIR contains faro_YYYY/ folders.")

    print(f"\n✓ Total records: {len(all_records):,}")

    # 6. Save intermediate JSON
    with open("newspaper_data.json", "w") as f:
        json.dump(all_records, f)
    print("✓ Saved newspaper_data.json")

    # 7. Generate the self-contained HTML visualization
    generate_html(all_records, OUTPUT_HTML)
    print(f"✓ Saved {OUTPUT_HTML}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: HTML generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_html(records: list[dict], output_path: str):
    """Embed records into the HTML template and write a self-contained file."""
    data_json = json.dumps(records)

    template_path = os.path.join(os.path.dirname(__file__), "map_template.html")
    if not os.path.exists(template_path):
        print(f"  ⚠ Template not found at {template_path}.")
        print("    Ensure map_template.html is in the same directory as this script.")
        return

    with open(template_path) as f:
        html = f.read()

    html = html.replace("/* __NEWSPAPER_DATA__ */", f"const NEWSPAPER_DATA = {data_json};")
    html = html.replace("__START_YEAR__", str(START_YEAR))
    html = html.replace("__END_YEAR__",   str(END_YEAR))

    with open(output_path, "w") as f:
        f.write(html)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="AmericanStories Hedonometer Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "By default the script loads newspaper_data.json (if it exists) and skips\n"
            "any years already present, so re-running the same range is always safe.\n"
            "Use --reprocess to force already-present years to be overwritten.\n\n"
            "Examples:\n"
            "  %(prog)s --start 1800 --end 1850\n"
            "  %(prog)s --start 1851 --end 1870          # extends existing data\n"
            "  %(prog)s --start 1800 --end 1850 --reprocess\n"
            "  %(prog)s --from-json                       # rebuild HTML only\n"
        ),
    )
    parser.add_argument(
        "--start", type=int, default=START_YEAR, metavar="YEAR",
        help=f"First year to process (default: {START_YEAR}).",
    )
    parser.add_argument(
        "--end", type=int, default=END_YEAR, metavar="YEAR",
        help=f"Last year to process (default: {END_YEAR}).",
    )
    parser.add_argument(
        "--sample-size", type=int, default=SAMPLE_SIZE, metavar="N",
        help=f"Maximum articles to sample per newspaper/year (default: {SAMPLE_SIZE}).",
    )
    parser.add_argument(
        "--min-scored-words", type=int, default=MIN_SCORED_WORDS, metavar="N",
        help=f"Minimum number of labMT-matched words a newspaper must have to be included "
             f"in the visualization (default: {MIN_SCORED_WORDS}). "
             f"Raising this filters out short or non-English publications.",
    )
    parser.add_argument(
        "--reprocess", action="store_true",
        help="Re-process and overwrite years that are already present in newspaper_data.json.",
    )
    parser.add_argument(
        "--from-json",
        metavar="PATH",
        nargs="?",
        const="newspaper_data.json",
        help="Skip data processing and regenerate the HTML directly from an existing "
             "JSON file (default: newspaper_data.json).",
    )
    args = parser.parse_args()

    START_YEAR       = args.start
    END_YEAR         = args.end
    SAMPLE_SIZE      = args.sample_size
    MIN_SCORED_WORDS = args.min_scored_words

    if args.from_json:
        # Regenerate HTML only – no new processing
        if not os.path.isfile(args.from_json):
            sys.exit(f"✗ JSON file not found: {args.from_json!r}")
        with open(args.from_json) as f:
            records = json.load(f)
        print(f"✓ Loaded {len(records):,} records from {args.from_json}")
        generate_html(records, OUTPUT_HTML)
        print(f"✓ Saved {OUTPUT_HTML}")

    else:
        if START_YEAR > END_YEAR:
            sys.exit(f"✗ --start ({START_YEAR}) must be <= --end ({END_YEAR})")
        # Load existing data unless --reprocess was given
        existing: list = []
        json_path = "newspaper_data.json"
        if not args.reprocess and os.path.isfile(json_path):
            with open(json_path) as f:
                existing = json.load(f)
            print(f"✓ Loaded {len(existing):,} existing records from {json_path}")
        elif args.reprocess:
            print("  --reprocess set: ignoring any existing data for the requested year range")
        main(existing_records=existing)
