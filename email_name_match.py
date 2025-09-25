"""Email and Name Matching Scoring Script.

This script analyses CSV records containing first name, last name, and email
addresses. It produces three objective-specific scores per record:

1. Name alignment score (local part of email vs. provided names and aliases).
2. Gender association score (first name and email keywords vs. reference lists).
3. Email quality score (bad keywords, disposable domains, and heuristic checks).

The script is designed for Windows command line usage but runs on any platform.
Default paths follow the project specification and can be overridden with CLI
flags. Reference files are auto-created with miniature fixtures if missing so
that an end-to-end smoke test can run immediately after installation.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from logging.handlers import RotatingFileHandler
import random
import re
import sys
import unicodedata
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple


# ---------------------------------------------------------------------------
# Default Windows-style paths (can be overridden via CLI arguments)
# ---------------------------------------------------------------------------
DEFAULT_INPUT_PATH = Path(r"C:\Adapt\Email Name Match")
DEFAULT_OUTPUT_PATH = Path(r"C:\Adapt\Email Name Match")
DEFAULT_NAMES_PATH = Path(r"C:\Adapt\Reference Files\All Names.csv")
DEFAULT_REFERENCE_JSON_PATH = Path(r"C:\Adapt\Email Name Match\reference_lists.json")
DEFAULT_FLAGGED_NAMES_PATH = Path(r"C:\Adapt\Email Name Match\flagged_names_report.csv")
DEFAULT_LOG_PATH = Path(r"C:\Adapt\Email Name Match\logs\email_name_match.log")


# ---------------------------------------------------------------------------
# Utility data classes
# ---------------------------------------------------------------------------
@dataclass
class NameAnalysis:
    score: int
    label: str
    details: List[str] = field(default_factory=list)


@dataclass
class GenderAnalysis:
    score: int
    label: str
    details: List[str] = field(default_factory=list)


@dataclass
class QualityAnalysis:
    score: int
    label: str
    deductions: List[str] = field(default_factory=list)


@dataclass
class RecordAnalysis:
    source_file: str
    line_number: int
    raw_row: Dict[str, str]
    name_analysis: NameAnalysis
    gender_analysis: GenderAnalysis
    quality_analysis: QualityAnalysis


# ---------------------------------------------------------------------------
# Progress indicator abstraction with graceful fallback
# ---------------------------------------------------------------------------
class ProgressReporter:
    """Wrapper around tqdm (if installed) with a minimal fallback."""

    def __init__(self, total: Optional[int] = None, description: str = "Processing") -> None:
        self._description = description
        self._total = total
        self._count = 0
        self._bar = None
        try:
            from tqdm import tqdm  # type: ignore

            self._bar = tqdm(total=total, desc=description, unit="row")
        except Exception:  # pragma: no cover - fallback path
            self._bar = None
            if total is not None and total > 0:
                logging.info("%s (0/%s)", description, total)

    def update(self, increment: int = 1) -> None:
        self._count += increment
        if self._bar is not None:
            self._bar.update(increment)
        else:
            if self._total:
                if self._count % max(1, self._total // 10) == 0:
                    logging.info("%s (%s/%s)", self._description, self._count, self._total)
            else:
                if self._count % 500 == 0:
                    logging.info("%s: %s rows processed", self._description, self._count)

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()
        else:
            logging.info("%s complete (%s rows)", self._description, self._count)


# ---------------------------------------------------------------------------
# File utility helpers
# ---------------------------------------------------------------------------
def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Fixture generation for missing reference or input files
# ---------------------------------------------------------------------------
def create_sample_reference_files(names_path: Path, reference_json_path: Path, flagged_path: Path) -> None:
    if not names_path.exists():
        ensure_parent_dir(names_path)
        sample_names = ["Name", "alice", "al", "bob", "bobby", "carol", "caroline", "dan", "danny", "lee", "kim"]
        with names_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            for value in sample_names:
                writer.writerow([value])

    if not reference_json_path.exists():
        ensure_parent_dir(reference_json_path)
        sample_reference = {
            "female_names": ["alice", "carol", "caroline", "kim"],
            "male_names": ["bob", "dan", "lee"],
            "bad_words": ["spam", "junk"],
            "last_names": ["smith", "johnson", "kim"],
            "disposable_email_domains": ["mailinator.com", "tempmail.test"],
            "aliases": {"al": ["albert", "alfred"], "bob": ["robert"]},
            "female_keyword": ["mom", "mrs", "queen"],
            "male_keyword": ["dad", "mr", "king"],
            "role_keywords": ["info", "sales", "support"],
        }
        with reference_json_path.open("w", encoding="utf-8") as f:
            json.dump(sample_reference, f, indent=2)

    if not flagged_path.exists():
        ensure_parent_dir(flagged_path)
        with flagged_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["name", "flags"])
            writer.writerow(["kim", "ambiguous"])
            writer.writerow(["lee", "ambiguous"])


def create_sample_input_if_missing(input_path: Path) -> None:
    ensure_directory(input_path)
    csv_candidates = list(input_path.glob("*.csv"))
    zip_candidates = list(input_path.glob("*.zip"))
    if csv_candidates or zip_candidates:
        return

    # Create a tiny CSV and ZIP so the script has data to crunch.
    sample_rows = [
        {
            "First Name": "Alice",
            "Last Name": "Smith",
            "Email": "alice.smith@example.com",
        },
        {
            "First Name": "Bob",
            "Last Name": "Lee",
            "Email": "robert.l@example.org",
        },
        {
            "First Name": "Carol",
            "Last Name": "Kim",
            "Email": "ckim@mailinator.com",
        },
        {
            "First Name": "Dan",
            "Last Name": "Brown",
            "Email": "info@unknown.test",
        },
    ]

    csv_path = input_path / "sample_input.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["First Name", "Last Name", "Email"])
        writer.writeheader()
        for row in sample_rows:
            writer.writerow(row)

    zip_path = input_path / "sample_input.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.write(csv_path, arcname="sample_input.csv")


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------
NAME_TOKEN_PATTERN = re.compile(r"[^a-z0-9]+")


def normalise_text(value: str) -> str:
    value = value or ""
    value = unicodedata.normalize("NFKD", value)
    value = value.encode("ascii", "ignore").decode("ascii", "ignore")
    return value.strip().lower()


def tokenise_local_part(local_part: str) -> List[str]:
    tokens = [token for token in NAME_TOKEN_PATTERN.split(local_part.lower()) if token]
    return tokens


def strip_non_alnum(value: str) -> str:
    return "".join(ch for ch in value if ch.isalnum())


def get_email_parts(email: str) -> Tuple[str, str]:
    if not email or "@" not in email:
        return "", ""
    local, _, domain = email.partition("@")
    return local.strip(), domain.strip()


def initial_of(value: str) -> str:
    value = value.strip()
    return value[0].lower() if value else ""


# ---------------------------------------------------------------------------
# Loading reference datasets
# ---------------------------------------------------------------------------
def load_all_names(names_path: Path) -> Set[str]:
    names: Set[str] = set()
    with names_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            names.add(normalise_text(row[0]))
    return names


def load_reference_lists(reference_json_path: Path) -> Dict[str, Sequence[str]]:
    with reference_json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    # normalise relevant lists for case-insensitive lookups
    normalised = {}
    for key, value in data.items():
        if isinstance(value, list):
            normalised[key] = [normalise_text(v) for v in value]
        elif isinstance(value, dict):
            normalised[key] = {normalise_text(k): [normalise_text(x) for x in v] for k, v in value.items()}
        else:
            normalised[key] = value
    return normalised


def load_flagged_names(flagged_path: Path) -> Set[str]:
    if not flagged_path.exists():
        return set()
    flagged: Set[str] = set()
    with flagged_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = normalise_text(row.get("name", ""))
            if name:
                flagged.add(name)
    return flagged


# ---------------------------------------------------------------------------
# Name analysis logic
# ---------------------------------------------------------------------------
ROLE_LOCAL_PARTS = {
    "admin",
    "alerts",
    "billing",
    "careers",
    "contact",
    "customerservice",
    "enquiries",
    "enquiry",
    "hello",
    "help",
    "hr",
    "info",
    "mail",
    "marketing",
    "newsletter",
    "noreply",
    "office",
    "orders",
    "sales",
    "service",
    "services",
    "support",
    "team",
}


NEUTRAL_NAME_SCORE = 3


def analyse_name_alignment(
    first_name: str,
    last_name: str,
    email: str,
    *,
    all_names: Set[str],
    alias_map: Dict[str, Set[str]],
    flagged_names: Set[str],
    extra_role_tokens: Set[str],
) -> NameAnalysis:
    first_clean = normalise_text(first_name)
    last_clean = normalise_text(last_name)
    local_part, domain_part = get_email_parts(normalise_text(email))

    local_tokens = tokenise_local_part(local_part)
    local_compact = strip_non_alnum(local_part)
    domain_tokens = tokenise_local_part(domain_part.replace(".", " "))

    first_aliases = {first_clean}
    last_aliases = {last_clean}

    if first_clean in alias_map:
        first_aliases.update(alias_map[first_clean])
    if last_clean in alias_map:
        last_aliases.update(alias_map[last_clean])

    details: List[str] = []
    score = 0
    positive_signal = False
    penalty = 0

    def token_matches(token: str, targets: Set[str]) -> bool:
        token_norm = normalise_text(token)
        return token_norm in targets

    first_token_hit = any(token_matches(tok, first_aliases) for tok in local_tokens)
    last_token_hit = any(token_matches(tok, last_aliases) for tok in local_tokens)
    first_partial = not first_token_hit and any(first_clean and first_clean in tok for tok in local_tokens)
    last_partial = not last_token_hit and any(last_clean and last_clean in tok for tok in local_tokens)

    initials = {initial_of(first_clean), initial_of(last_clean)} - {""}
    first_initial_hit = any(tok == initial_of(first_clean) for tok in local_tokens if len(tok) == 1)
    last_initial_hit = any(tok == initial_of(last_clean) for tok in local_tokens if len(tok) == 1)

    joined_variants = {
        first_clean + last_clean,
        last_clean + first_clean,
    }
    if first_clean and last_clean:
        if local_compact in joined_variants:
            score += 40
            positive_signal = True
            details.append("Local part is exact full-name concatenation")
        elif any(
            local_compact.startswith(variant) or local_compact.endswith(variant) for variant in joined_variants
        ):
            score += 25
            positive_signal = True
            details.append("Local part contains full-name concatenation")

    if first_token_hit and last_token_hit:
        score += 60
        positive_signal = True
        details.append("Local tokens include both first and last name")
    else:
        if first_token_hit:
            score += 30
            positive_signal = True
            details.append("Local tokens include the first name")
        if last_token_hit:
            score += 30
            positive_signal = True
            details.append("Local tokens include the last name")

    if first_partial:
        score += 12
        positive_signal = True
        details.append("Partial match to first name inside local tokens")
    if last_partial:
        score += 12
        positive_signal = True
        details.append("Partial match to last name inside local tokens")

    if first_initial_hit and last_initial_hit:
        score += 18
        positive_signal = True
        details.append("Both initials present in local tokens")
    else:
        if first_initial_hit:
            score += 10
            positive_signal = True
            details.append("First-name initial present in local tokens")
        if last_initial_hit:
            score += 10
            positive_signal = True
            details.append("Last-name initial present in local tokens")

    # Other names detection (excluding flagged ambiguous names and provided names)
    other_name_tokens = []
    for token in local_tokens:
        token_norm = normalise_text(token)
        if not token_norm or token_norm in first_aliases or token_norm in last_aliases:
            continue
        if token_norm in flagged_names:
            details.append(f"Token '{token}' ignored (flagged ambiguous name)")
            continue
        if token_norm in all_names:
            other_name_tokens.append(token)

    if other_name_tokens:
        penalty += 25
        details.append("Local part contains additional name(s): " + ", ".join(other_name_tokens))

    # Role-based detection uses both local and domain tokens
    combined_role_tokens = ROLE_LOCAL_PARTS | extra_role_tokens
    role_hits = {tok for tok in local_tokens + domain_tokens if tok in combined_role_tokens}
    if role_hits:
        penalty += 35
        details.append("Role-based tokens detected: " + ", ".join(sorted(role_hits)))

    if not local_tokens and not local_part:
        details.append("Email missing local part; treated as neutral name match")

    if penalty:
        details.append(f"Total penalty applied: -{penalty} points")

    score = max(0, score - penalty)
    score = min(100, score)

    if not positive_signal and penalty == 0:
        score = NEUTRAL_NAME_SCORE
        label = "Neutral"
        details.append("No definitive alignment between email and provided names")
    elif score >= 70:
        label = "Strong match"
    elif score >= 35:
        label = "Partial match"
    elif penalty > 0:
        label = "Weak or conflicting"
    else:
        label = "Weak match"

    return NameAnalysis(score=int(round(score)), label=label, details=details)


# ---------------------------------------------------------------------------
# Gender analysis logic
# ---------------------------------------------------------------------------
NEUTRAL_GENDER_SCORE = 5


def analyse_gender(
    first_name: str,
    email: str,
    *,
    female_names: Set[str],
    male_names: Set[str],
    female_keywords: Set[str],
    male_keywords: Set[str],
) -> GenderAnalysis:
    first_clean = normalise_text(first_name)
    local_part, _ = get_email_parts(normalise_text(email))
    local_tokens = tokenise_local_part(local_part)

    score = 0
    details: List[str] = []
    has_signal = False

    female_name_hit = first_clean in female_names
    male_name_hit = first_clean in male_names

    if female_name_hit:
        score += 40
        has_signal = True
        details.append("First name matches female reference list")
    if male_name_hit:
        score -= 40
        has_signal = True
        details.append("First name matches male reference list")

    female_keyword_hits = [tok for tok in local_tokens if tok in female_keywords]
    male_keyword_hits = [tok for tok in local_tokens if tok in male_keywords]

    if female_keyword_hits:
        score += 20
        has_signal = True
        details.append("Female keyword(s) in email: " + ", ".join(female_keyword_hits))
    if male_keyword_hits:
        score -= 20
        has_signal = True
        details.append("Male keyword(s) in email: " + ", ".join(male_keyword_hits))

    label: str
    if not has_signal:
        score = NEUTRAL_GENDER_SCORE
        label = "Neutral"
        details.append("No gender indicators detected")
    elif score >= 30:
        label = "Female leaning"
    elif score <= -30:
        label = "Male leaning"
    else:
        label = "Ambiguous"

    return GenderAnalysis(score=int(score), label=label, details=details)


# ---------------------------------------------------------------------------
# Quality analysis logic
# ---------------------------------------------------------------------------
BAD_EXACT_LOCAL_PARTS = {
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "none",
    "na",
    "noemail",
    "email",
    "me",
    "123",
    "abc",
    "asdf",
    "test",
    "unknown",
    "n/a",
    "invalidaddress",
    "refused",
    "declined",
    "withheld",
    "noreply",
    "nope",
}

BAD_EXACT_EMAILS = {
    "email@email",
    "gmail@gmail.com",
    "me@me.com",
    "test@",
    "@test",
    "no.email@",
    "@no.email",
    "@unknown",
    "unknown@",
    "notprovided",
    "nonegiven",
    "notavailable",
}


def analyse_quality(
    email: str,
    *,
    bad_words: Set[str],
    disposable_domains: Set[str],
) -> QualityAnalysis:
    email_norm = normalise_text(email)
    local_part, domain_part = get_email_parts(email_norm)
    local_tokens = tokenise_local_part(local_part)
    domain_lower = domain_part.lower()

    score = 100
    deductions: List[str] = []

    if not local_part:
        score -= 40
        deductions.append("Missing local part before '@'")

    if email_norm in BAD_EXACT_EMAILS:
        score -= 40
        deductions.append("Email matches known invalid placeholder")

    if local_part in BAD_EXACT_LOCAL_PARTS:
        score -= 35
        deductions.append("Local part matches invalid placeholder token")

    if local_part and local_part.isdigit():
        score -= 15
        deductions.append("Local part is digits only")

    if local_part and len(set(local_part)) == 1 and len(local_part) >= 5:
        score -= 20
        deductions.append("Local part is a single repeated character")

    if local_part and re.search(r"([a-z0-9])\1{4,}", local_part):
        score -= 15
        deductions.append("Same character repeated 5+ times in local part")

    if local_part and not re.search(r"[a-z]", local_part):
        score -= 10
        deductions.append("No alphabetic characters in local part")

    for bad_word in bad_words:
        if bad_word and bad_word in email_norm:
            score -= 10
            deductions.append(f"Contains bad word '{bad_word}'")

    if domain_lower:
        for disposable in disposable_domains:
            if disposable and domain_lower.endswith(disposable):
                score -= 25
                deductions.append(f"Disposable domain detected ({disposable})")

    if not domain_lower:
        score -= 20
        deductions.append("Missing domain part after '@'")

    if local_tokens and any(len(tok) == 1 for tok in local_tokens):
        score -= 5
        deductions.append("Single-character token inside local part")

    score = max(0, score)

    if score >= 80:
        label = "High"
    elif score >= 50:
        label = "Moderate"
    else:
        label = "Low"

    return QualityAnalysis(score=score, label=label, deductions=deductions)


# ---------------------------------------------------------------------------
# CSV processing helpers
# ---------------------------------------------------------------------------
def detect_header_fields(header: Sequence[str]) -> Dict[str, str]:
    mapping = {name.lower().strip(): name for name in header}
    field_map: Dict[str, str] = {}
    for target in ("First Name", "Last Name", "Email"):
        lookup = target.lower()
        if lookup in mapping:
            field_map[target] = mapping[lookup]
        else:
            for candidate_lower, candidate_original in mapping.items():
                simplified = candidate_lower.replace("_", " ").replace("-", " ")
                if lookup.replace(" ", "") == simplified.replace(" ", ""):
                    field_map[target] = candidate_original
                    break
    missing = [key for key in ("First Name", "Last Name", "Email") if key not in field_map]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")
    return field_map


def iter_csv_rows(source_name: str, file_obj: Iterable[str]) -> Iterator[Tuple[int, Dict[str, str]]]:
    reader = csv.DictReader(file_obj)
    if reader.fieldnames is None:
        raise ValueError(f"File '{source_name}' is missing a header row")
    field_map = detect_header_fields(reader.fieldnames)

    for line_number, row in enumerate(reader, start=2):
        try:
            first = row.get(field_map["First Name"], "")
            last = row.get(field_map["Last Name"], "")
            email = row.get(field_map["Email"], "")
        except KeyError as exc:  # pragma: no cover - defensive guard
            raise ValueError(f"Unexpected missing column while reading '{source_name}'") from exc

        yield line_number, {"First Name": first, "Last Name": last, "Email": email}


def iter_input_records(input_path: Path) -> Iterator[Tuple[str, int, Dict[str, str]]]:
    if input_path.is_file():
        yield from _iter_file_records(input_path)
        return

    files = sorted(list(input_path.glob("*.csv")) + list(input_path.glob("*.zip")))
    for file_path in files:
        yield from _iter_file_records(file_path)


def _iter_file_records(file_path: Path) -> Iterator[Tuple[str, int, Dict[str, str]]]:
    if file_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(file_path, "r") as zip_file:
            for member in zip_file.infolist():
                if not member.filename.lower().endswith(".csv"):
                    continue
                with zip_file.open(member, "r") as member_file:
                    text_stream = (line.decode("utf-8", "ignore") for line in member_file)
                    try:
                        for line_number, row in iter_csv_rows(
                            f"{file_path.name}:{member.filename}", text_stream
                        ):
                            yield file_path.name + ":" + member.filename, line_number, row
                    except ValueError as exc:
                        logging.warning("Skipping %s:%s (%s)", file_path.name, member.filename, exc)
    elif file_path.suffix.lower() == ".csv":
        with file_path.open("r", encoding="utf-8", newline="") as f:
            try:
                for line_number, row in iter_csv_rows(file_path.name, f):
                    yield file_path.name, line_number, row
            except ValueError as exc:
                logging.warning("Skipping %s (%s)", file_path.name, exc)


# ---------------------------------------------------------------------------
# Main processing routine
# ---------------------------------------------------------------------------
def process_records(
    records: Iterator[Tuple[str, int, Dict[str, str]]],
    *,
    all_names: Set[str],
    reference_lists: Dict[str, Sequence[str]],
    flagged_names: Set[str],
    progress_description: str,
) -> Iterator[RecordAnalysis]:
    alias_map_raw = reference_lists.get("aliases", {})
    alias_map: Dict[str, Set[str]] = {}
    if isinstance(alias_map_raw, dict):
        for key, values in alias_map_raw.items():
            key_norm = normalise_text(key)
            if key_norm in flagged_names:
                continue
            if isinstance(values, (list, tuple, set)):
                filtered = {normalise_text(v) for v in values} - flagged_names
                if filtered:
                    alias_map[key_norm] = filtered


    female_names = set(reference_lists.get("female_names", []))
    male_names = set(reference_lists.get("male_names", []))
    female_keywords = set(reference_lists.get("female_keyword", []))
    male_keywords = set(reference_lists.get("male_keyword", []))
    bad_words = set(reference_lists.get("bad_words", []))
    disposable_domains = set(reference_lists.get("disposable_email_domains", []))
    extra_role_tokens = set(reference_lists.get("role_keywords", []))

    if flagged_names:
        female_names -= flagged_names
        male_names -= flagged_names
        extra_role_tokens -= flagged_names

    progress = ProgressReporter(description=progress_description)
    try:
        for source_name, line_number, row in records:
            name_analysis = analyse_name_alignment(
                row.get("First Name", ""),
                row.get("Last Name", ""),
                row.get("Email", ""),
                all_names=all_names,
                alias_map=alias_map,
                flagged_names=flagged_names,
                extra_role_tokens=extra_role_tokens,
            )

            gender_analysis = analyse_gender(
                row.get("First Name", ""),
                row.get("Email", ""),
                female_names=female_names,
                male_names=male_names,
                female_keywords=female_keywords,
                male_keywords=male_keywords,
            )

            quality_analysis = analyse_quality(
                row.get("Email", ""),
                bad_words=bad_words,
                disposable_domains=disposable_domains,
            )

            progress.update()
            yield RecordAnalysis(
                source_file=source_name,
                line_number=line_number,
                raw_row=row,
                name_analysis=name_analysis,
                gender_analysis=gender_analysis,
                quality_analysis=quality_analysis,
            )
    finally:
        progress.close()


def write_results(output_path: Path, analyses: Iterable[RecordAnalysis]) -> Path:
    ensure_directory(output_path.parent)
    temp_path = output_path.with_suffix(".tmp")
    with temp_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "First Name",
            "Last Name",
            "Email",
            "Source",
            "RowNumber",
            "NameMatchScore",
            "NameMatchLabel",
            "NameMatchDetails",
            "GenderScore",
            "GenderLabel",
            "GenderDetails",
            "QualityScore",
            "QualityLabel",
            "QualityDeductions",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for analysis in analyses:
            writer.writerow(
                {
                    "First Name": analysis.raw_row.get("First Name", ""),
                    "Last Name": analysis.raw_row.get("Last Name", ""),
                    "Email": analysis.raw_row.get("Email", ""),
                    "Source": analysis.source_file,
                    "RowNumber": analysis.line_number,
                    "NameMatchScore": analysis.name_analysis.score,
                    "NameMatchLabel": analysis.name_analysis.label,
                    "NameMatchDetails": " | ".join(analysis.name_analysis.details),
                    "GenderScore": analysis.gender_analysis.score,
                    "GenderLabel": analysis.gender_analysis.label,
                    "GenderDetails": " | ".join(analysis.gender_analysis.details),
                    "QualityScore": analysis.quality_analysis.score,
                    "QualityLabel": analysis.quality_analysis.label,
                    "QualityDeductions": " | ".join(analysis.quality_analysis.deductions),
                }
            )
    temp_path.replace(output_path)
    return output_path


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
def setup_logging(log_path: Path, verbose: bool = False) -> None:
    ensure_parent_dir(log_path)
    handlers: List[logging.Handler] = []

    file_handler = RotatingFileHandler(log_path, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    handlers.append(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    handlers.append(stream_handler)

    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, handlers=handlers)
    logging.debug("Logging initialised. Verbose=%s", verbose)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyse email/name alignment, gender indicators, and email quality.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH, help="Input directory or file path")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Directory for output CSV")
    parser.add_argument(
        "--names-csv", type=Path, default=DEFAULT_NAMES_PATH, help="Path to All Names CSV reference"
    )
    parser.add_argument(
        "--reference-json", type=Path, default=DEFAULT_REFERENCE_JSON_PATH, help="Path to reference_lists.json"
    )
    parser.add_argument(
        "--flagged-names", type=Path, default=DEFAULT_FLAGGED_NAMES_PATH, help="Path to flagged names CSV"
    )
    parser.add_argument(
        "--log-file", type=Path, default=DEFAULT_LOG_PATH, help="Log file path (rotating logs are used)"
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="email_name_match_scores.csv",
        help="Name of the output CSV file",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main execution entry point
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_arguments(argv)

    try:
        if args.input.is_dir():
            ensure_directory(args.input)
        else:
            ensure_directory(args.input.parent)
        ensure_directory(args.output)
        create_sample_reference_files(args.names_csv, args.reference_json, args.flagged_names)
        if args.input.is_dir():
            create_sample_input_if_missing(args.input)
        setup_logging(args.log_file, verbose=args.verbose)

        all_names = load_all_names(args.names_csv)
        reference_lists = load_reference_lists(args.reference_json)
        flagged_names = load_flagged_names(args.flagged_names)

        logging.info("Loaded %s reference names, %s female names, %s male names", len(all_names), len(reference_lists.get("female_names", [])), len(reference_lists.get("male_names", [])))

        output_file = args.output / args.output_name
        records_iterator = iter_input_records(args.input)
        analyses_iterator = process_records(
            records_iterator,
            all_names=all_names,
            reference_lists=reference_lists,
            flagged_names=flagged_names,
            progress_description="Evaluating records",
        )

        result_path = write_results(output_file, analyses_iterator)
        logging.info("Analysis complete. Results written to %s", result_path)
        return 0
    except Exception as exc:
        logging.exception("Processing failed: %s", exc)
        return 1


if __name__ == "__main__":
    random.seed(1337)
    sys.exit(main())

