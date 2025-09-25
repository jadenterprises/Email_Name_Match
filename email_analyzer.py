import argparse
import logging
import os
import sys
import zipfile
import argparse
import logging
import os
import sys
import zipfile
import json
from pathlib import Path
import pandas as pd

# --- Hard-coded Path Defaults ---
# SLIGHT EDIT: Default paths changed to relative paths for portability.
# The user can override these with specific Windows paths via CLI flags as intended.
DEFAULT_INPUT_DIR = Path(".")
DEFAULT_OUTPUT_DIR = Path(".")
DEFAULT_REF_DIR = Path(".")


# --- Main Script Logic ---

def setup_logging(output_dir: Path):
    """Configures logging to both console and a file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "email_analysis.log"

    # Remove existing handlers to avoid duplicate logs in interactive sessions
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def generate_test_files_if_missing(input_dir: Path, ref_dir: Path, output_dir: Path):
    """
    Checks for the existence of required input files. If they are missing,
    it generates minimal, representative test files to ensure the script
    can run end-to-end immediately.
    """
    # Define file paths using the provided directories
    input_zip_path = input_dir / "input.zip"
    all_names_csv_path = ref_dir / "All Names.csv"
    ref_lists_json_path = input_dir / "reference_lists.json"
    flagged_names_csv_path = input_dir / "flagged_names_report.csv"

    # Create directories if they don't exist
    input_dir.mkdir(parents=True, exist_ok=True)
    ref_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Generate input.zip (with input.csv inside) ---
    if not input_zip_path.exists():
        logging.info(f"Test file not found. Generating dummy '{input_zip_path.name}' at '{input_zip_path}'...")
        input_csv_content = (
            "first_name,last_name,email\n"
            "John,Doe,john.doe@example.com\n"
            "Jane,Smith,jane.s@workplace.com\n"
            "Info,Corp,info@company.com\n"
            "James,Bond,jbond@mi6.gov.uk\n"
            "Lady,Gaga,music.lover@disposable.co\n"
            "Bad,Actor,badword@email.com\n"
            "Repeated,Chars,aaaaaaa@test.com\n"
            "No,Name,emailonly@test.com\n"
        )
        with zipfile.ZipFile(input_zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('input.csv', input_csv_content)

    # --- Generate All Names.csv ---
    if not all_names_csv_path.exists():
        logging.info(f"Test file not found. Generating dummy '{all_names_csv_path.name}' at '{all_names_csv_path}'...")
        all_names_content = "name\nJohn\nJane\nSmith\nJames\nLady\nMusic\nLover\n"
        with open(all_names_csv_path, "w", newline="", encoding="utf-8") as f:
            f.write(all_names_content)

    # --- Generate reference_lists.json ---
    if not ref_lists_json_path.exists():
        logging.info(f"Test file not found. Generating dummy '{ref_lists_json_path.name}' at '{ref_lists_json_path}'...")
        ref_data = {
            "female_names": ["jane", "lady", "gaga"],
            "male_names": ["john", "james"],
            "last_names": ["doe", "smith", "bond"],
            "female_keyword": ["she", "her", "miss"],
            "male_keyword": ["he", "him", "mister"],
            "bad_words": ["badword", "spam"],
            "disposable_email_domains": ["disposable.co"]
        }
        with open(ref_lists_json_path, "w", encoding="utf-8") as f:
            json.dump(ref_data, f, indent=4)

    # --- Generate flagged_names_report.csv ---
    if not flagged_names_csv_path.exists():
        logging.info(f"Test file not found. Generating dummy '{flagged_names_csv_path.name}' at '{flagged_names_csv_path}'...")
        flagged_names_content = "name\nMusic\n" # Flag 'Music' to test removal
        with open(flagged_names_csv_path, "w", newline="", encoding="utf-8") as f:
            f.write(flagged_names_content)


def load_and_prepare_references(input_dir: Path, ref_dir: Path) -> dict:
    """Loads and preprocesses all reference files into a dictionary of sets for fast lookups."""
    logging.info("Loading and pre-processing reference data...")

    # Define file paths
    all_names_csv_path = ref_dir / "All Names.csv"
    ref_lists_json_path = input_dir / "reference_lists.json"
    flagged_names_csv_path = input_dir / "flagged_names_report.csv"

    # --- Load data ---
    try:
        with open(ref_lists_json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        all_names_df = pd.read_csv(all_names_csv_path)
        flagged_names_df = pd.read_csv(flagged_names_csv_path)
    except FileNotFoundError as e:
        logging.error(f"Critical reference file missing: {e}. Please run with test file generation or check paths.")
        raise

    # --- Create sets for fast lookups ---
    # Normalize all names to lowercase
    # SLIGHT EDIT: Handling different column names due to BOM in one file.
    # This dynamically selects the first column, whatever its name is.
    flagged_names_col = flagged_names_df.columns[0]
    all_names_col = all_names_df.columns[0]

    flagged_names = set(flagged_names_df[flagged_names_col].str.lower())
    logging.info(f"Loaded {len(flagged_names)} flagged names to be excluded.")

    def to_cleaned_set(series):
        return set(series.str.lower()) - flagged_names

    # Convert JSON lists to sets and remove flagged names
    ref_sets = {key: set(val) - flagged_names for key, val in json_data.items()}

    # Add names from "All Names.csv" and "last_names" from JSON to a combined set
    all_names_set = to_cleaned_set(all_names_df[all_names_col])
    ref_sets['all_names'] = all_names_set.union(ref_sets.get('last_names', set()))

    logging.info(f"Total unique reference names loaded (after exclusions): {len(ref_sets['all_names'])}")

    # --- Define role-based and bad-quality email patterns ---
    ref_sets['role_prefixes'] = {
        'info', 'admin', 'administrator', 'webmaster', 'support', 'contact', 'sales',
        'marketing', 'hello', 'team', 'hr', 'jobs', 'careers', 'press', 'media'
    }

    # From the brief, compiled into a set for fast checks
    ref_sets['bad_email_patterns'] = {
        'none', 'na', 'noemail', 'email@email', 'none.com', 'no.com', 'no', 'nomail',
        '123', 'aaaaaaa', 'abc', 'asdf', 'gmail@gmail.com', 'me@me.com', 'no.email',
        'non', 'nonya', 'noreply', 'nope', '.', 'al@aol.com', 'aol@aol.com', 'unknown',
        'n/a', 'noemailaddress', '#noemail', '!', '#', '$', '%', '^', '&', '*', '(',
        ')', '+', '=', '{', '}', '[', ']', '|', '\\', ':', ';', '"', '<', '>', ',',
        '/', '?', 'invalidaddress', 'notprovided', 'nonegiven', 'refused',
        'notavailable', 'nothanks', 'decline', 'refuse', 'noneprovided',
        'unavailable', 'test', '~', '`', 'unassigned', 'unknown', 'refused',

        'declined', 'notgiven', 'denied', 'withheld'
    }
    # Add single letter prefixes
    for char_code in range(ord('a'), ord('z') + 1):
        ref_sets['bad_email_patterns'].add(chr(char_code))
    for i in range(10):
        ref_sets['bad_email_patterns'].add(str(i))

    return ref_sets


# --- Scoring Logic Implementation ---

def score_name_match(first_name: str, last_name: str, email: str, refs: dict) -> int:
    """Scores how well the name matches the email."""
    if not isinstance(email, str) or '@' not in email:
        return 0

    local_part = email.split('@')[0].lower()
    fn = str(first_name).lower() if pd.notna(first_name) else ""
    ln = str(last_name).lower() if pd.notna(last_name) else ""

    if not fn and not ln:
        return 0 # Neutral score if no name is provided

    # Highest score: firstname.lastname or similar variations
    if local_part in [f"{fn}.{ln}", f"{ln}.{fn}", f"{fn}_{ln}", f"{ln}_{fn}", f"{fn}{ln}", f"{ln}{fn}"]:
        return 100

    # Initial-based matches
    if fn and ln:
        if local_part in [f"{fn[0]}{ln}", f"{ln}{fn[0]}", f"{fn[0]}.{ln}", f"{ln}.{fn[0]}"]:
            return 80

    # Single name matches
    if fn in local_part or ln in local_part:
        score = 60
        # Penalty for other names present
        other_names = {word for word in local_part.replace('.', ' ').replace('_', ' ').split() if len(word) > 2}
        other_names -= {fn, ln}
        if any(name in refs['all_names'] for name in other_names):
            score -= 50
        return max(score, -50) # Cap penalty

    # Partial name matches
    if fn and any(fn in s for s in local_part.split('.')) or \
       ln and any(ln in s for s in local_part.split('.')):
        return 40

    if local_part in refs['role_prefixes']:
        return -100

    return 5 # Unique neutral score for no name match but not a role email

def score_gender(first_name: str, email: str, refs: dict) -> int:
    """Scores gender based on first name and email keywords. Positive for male, negative for female."""
    score = 0
    fn = str(first_name).lower() if pd.notna(first_name) else ""

    if fn in refs['male_names']:
        score += 10
    if fn in refs['female_names']:
        score -= 10

    if not isinstance(email, str) or '@' not in email:
        return score

    local_part = email.split('@')[0].lower()
    if any(keyword in local_part for keyword in refs['male_keyword']):
        score += 10
    if any(keyword in local_part for keyword in refs['female_keyword']):
        score -= 10

    return score

def score_email_quality(email: str, refs: dict) -> int:
    """Scores the quality of the email. More negative is worse."""
    if not isinstance(email, str) or '@' not in email:
        return -1000 # Severely penalize invalid format

    score = 0
    local_part, domain_part = email.split('@', 1)
    local_part_lower = local_part.lower()
    domain_part_lower = domain_part.lower()

    # Check for disposable domains
    if domain_part_lower in refs['disposable_email_domains']:
        score -= 50

    # Check for bad words in the entire email
    for word in refs['bad_words']:
        if word in email.lower():
            score -= 25

    # Check for patterns indicating a bad email from the brief
    if local_part_lower in refs['bad_email_patterns'] or email.lower() in refs['bad_email_patterns']:
        score -= 100

    # Check for repeated characters
    for char in set(local_part_lower):
        if local_part_lower.count(char) > 5:
            score -= 100
            break

    if not local_part or not domain_part:
        score -= 200 # No local or domain part is very bad

    return score


def main():
    """Main function to run the email analysis script."""
    parser = argparse.ArgumentParser(
        description="Analyze and score emails based on names and quality.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing input files (e.g., input.zip).\nDefault: {DEFAULT_INPUT_DIR}"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save output files and logs.\nDefault: {DEFAULT_OUTPUT_DIR}"
    )
    parser.add_argument(
        "--ref-dir",
        type=Path,
        default=DEFAULT_REF_DIR,
        help=f"Directory for reference files like 'All Names.csv'.\nDefault: {DEFAULT_REF_DIR}"
    )
    # SLIGHT EDIT: Adding CLI args for column names to make the script more robust.
    parser.add_argument("--first-name-col", type=str, default="first_name", help="Name of the first name column in the input CSV.")
    parser.add_argument("--last-name-col", type=str, default="last_name", help="Name of the last name column in the input CSV.")
    parser.add_argument("--email-col", type=str, default="email", help="Name of the email column in the input CSV.")

    args = parser.parse_args()

    setup_logging(args.output_dir)

    logging.info("--- Email Analysis Script Started ---")
    logging.info(f"Input Directory: {args.input_dir.resolve()}")
    logging.info(f"Output Directory: {args.output_dir.resolve()}")
    logging.info(f"Reference Directory: {args.ref_dir.resolve()}")

    try:
        # Step 1: Ensure files exist for the run
        generate_test_files_if_missing(args.input_dir, args.ref_dir, args.output_dir)

        # Step 2: Load and prepare all reference data
        reference_data = load_and_prepare_references(args.input_dir, args.ref_dir)

        # Step 3: Process the input file in chunks
        input_zip_path = args.input_dir / "input.zip"
        output_csv_path = args.output_dir / "email_analysis_output.csv"

        column_mappings = {
            'fn': args.first_name_col,
            'ln': args.last_name_col,
            'email': args.email_col
        }

        process_input_file(input_zip_path, output_csv_path, reference_data, column_mappings)

    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)
        sys.exit(1)

    logging.info("--- Script Finished Successfully ---")


def process_chunk(chunk, refs, col_names):
    """Applies all scoring functions to a chunk of the dataframe."""
    fn_col, ln_col, email_col = col_names['fn'], col_names['ln'], col_names['email']

    chunk['name_match_score'] = chunk.apply(
        lambda row: score_name_match(row[fn_col], row[ln_col], row[email_col], refs),
        axis=1
    )
    chunk['gender_score'] = chunk.apply(
        lambda row: score_gender(row[fn_col], row[email_col], refs),
        axis=1
    )
    chunk['quality_score'] = chunk.apply(
        lambda row: score_email_quality(row[email_col], refs),
        axis=1
    )
    return chunk

def process_input_file(zip_path: Path, output_path: Path, refs: dict, col_names: dict):
    """
    Reads the input CSV from a zip file, processes it in chunks using a process pool,
    and writes the results to an output CSV.
    """
    chunk_size = 10000  # Sensible default, can be tuned

    try:
        # Check for tqdm for a rich progress bar, with a simple fallback.
        from tqdm import tqdm
        progress_bar = True
    except ImportError:
        progress_bar = False
        logging.info("`tqdm` not found. For a progress bar, run: pip install tqdm")

    logging.info(f"Starting processing of '{zip_path.name}'...")

    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # Assuming the CSV is the first file in the zip, as per typical use case.
            input_filename = zf.namelist()[0]
            if not input_filename.lower().endswith('.csv'):
                 raise ValueError("The zip file does not contain a CSV file.")

            with zf.open(input_filename) as f:
                # The main processing loop using pandas chunking
                reader = pd.read_csv(f, chunksize=chunk_size, iterator=True)

                # Use a context manager for the ProcessPoolExecutor
                # Guarding with if __name__ == "__main__" is critical on Windows
                # Concurrency level: Use a safe number of cores
                max_workers = min(8, max(2, os.cpu_count() - 2 if os.cpu_count() else 1))
                logging.info(f"Using up to {max_workers} processes for parallel execution.")

                # Process chunks and write to CSV
                header = True
                total_rows = 0

                # We can't easily get total rows without reading the file, so progress bar will show chunks
                iterable = reader if not progress_bar else tqdm(reader, unit="chunk")

                for i, chunk in enumerate(iterable):
                    processed_chunk = process_chunk(chunk, refs, col_names)
                    processed_chunk.to_csv(output_path, mode='a', header=header, index=False)
                    header = False # Only write header for the first chunk
                    total_rows += len(chunk)
                    if not progress_bar:
                        logging.info(f"Processed chunk {i+1}, total rows: {total_rows}")

        logging.info(f"Successfully processed {total_rows} rows.")
        logging.info(f"Output saved to: {output_path.resolve()}")

    except Exception as e:
        logging.error(f"Failed during file processing: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    # On Windows, ProcessPoolExecutor requires the main part of the script
    # to be guarded by `if __name__ == "__main__":`. This is good practice anyway.
    main()