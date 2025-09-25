import argparse
import logging
import os
import sys
import zipfile

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


    """
    Checks for the existence of required input files. If they are missing,
    it generates minimal, representative test files to ensure the script
    can run end-to-end immediately.
    """

    input_zip_path = input_dir / "input.zip"
    all_names_csv_path = ref_dir / "All Names.csv"
    ref_lists_json_path = input_dir / "reference_lists.json"
    flagged_names_csv_path = input_dir / "flagged_names_report.csv"

    # Create directories if they don't exist
    input_dir.mkdir(parents=True, exist_ok=True)
    ref_dir.mkdir(parents=True, exist_ok=True)


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


    args = parser.parse_args()

    setup_logging(args.output_dir)

    logging.info("--- Email Analysis Script Started ---")
    logging.info(f"Input Directory: {args.input_dir.resolve()}")
    logging.info(f"Output Directory: {args.output_dir.resolve()}")
    logging.info(f"Reference Directory: {args.ref_dir.resolve()}")

    try:


    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)
        sys.exit(1)

    logging.info("--- Script Finished Successfully ---")



if __name__ == "__main__":
    # On Windows, ProcessPoolExecutor requires the main part of the script
    # to be guarded by `if __name__ == "__main__":`. This is good practice anyway.
    main()