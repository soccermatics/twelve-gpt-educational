import json
import re
import pandas as pd
import gspread
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download("punkt")
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("punkt_tab")

import os

# --- Configuration ---
ttype = "country"
JSON_FILE_PATH = f"C:/Users/Amy/Desktop/Green_Git/twelve-gpt-educational/evaluation/2025-04-29/prompt_v1_{ttype}/data_points.json"  # <--- CHANGE THIS TO YOUR JSON FILE NAME
SPREADSHEET_NAME = f"2025-04-05_{ttype}"  # <--- Name for the new Google Sheet
# WORKSHEET_NAME = f"{ttype}"
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.file",
]
CREDENTIALS_FILE = (
    ".streamlit/client_secret.json"  # Downloaded from Google Cloud Console
)
TOKEN_FILE = ".streamlit/token.json"
BLEU_SCORE_THRESHOLD = 0.005  # Adjust as needed - lower means more lenient matching

# --- Helper Functions ---


def authenticate_gsheet():
    """Authenticates with Google Sheets API and returns the client."""
    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception as e:
                print(f"Error refreshing token: {e}. Re-authenticating...")
                if os.path.exists(TOKEN_FILE):
                    os.remove(TOKEN_FILE)  # Remove invalid token
                flow = InstalledAppFlow.from_client_secrets_file(
                    CREDENTIALS_FILE, SCOPES
                )
                creds = flow.run_local_server(port=0)
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open(TOKEN_FILE, "w") as token:
            token.write(creds.to_json())

    try:
        client = gspread.authorize(creds)
        return client
    except Exception as e:
        print(f"Error authorizing gspread client: {e}")
        # Attempt to delete token and re-authenticate if authorization fails
        if os.path.exists(TOKEN_FILE):
            print(
                "Deleting potentially corrupt token.json and re-running authentication..."
            )
            os.remove(TOKEN_FILE)
            return authenticate_gsheet()  # Recursive call to try again
        else:
            raise  # Re-raise the exception if token deletion didn't help


def split_into_sentences(text):
    """Splits text into sentences using NLTK."""
    if not text or not isinstance(text, str):
        return []
    # Basic cleaning: replace carriage returns, excessive newlines/spaces
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n\s*\n", "\n", text)  # Replace multiple newlines with one
    text = re.sub(r"\s+", " ", text).strip()  # Replace multiple spaces with one
    try:
        sentences = nltk.sent_tokenize(text)
        # Further split if sentences seem merged by ``` or similar artifacts
        final_sentences = []
        for sent in sentences:
            # Split based on common separators if they appear mid-sentence unusually
            parts = re.split(
                r"(?<!\w)\.(?!\w)|(?<=\S)\s*[`]{3,}\s*(?=\S)|(?<=\S)\s*\n\s*(?=\S)",
                sent,
            )
            final_sentences.extend([p.strip() for p in parts if p and p.strip()])
        return [
            s for s in final_sentences if len(s.split()) > 2
        ]  # Filter out very short/empty strings
    except Exception as e:
        print(f"Error tokenizing text: {text[:100]}... Error: {e}")
        return [text]  # Return original text as a single sentence on error


def calculate_bleu(reference_tokens, candidate_tokens):
    """Calculates BLEU score between a candidate sentence and a reference (factor name)."""
    if not candidate_tokens or not reference_tokens:
        return 0.0
    # BLEU expects a list of reference lists
    reference_list = [reference_tokens]
    # Use smoothing function for short sentences/references
    smoothing = SmoothingFunction().method1
    try:
        score = sentence_bleu(
            reference_list, candidate_tokens, smoothing_function=smoothing
        )
        return score
    except ZeroDivisionError:
        return 0.0  # Handle cases where no n-grams match
    except Exception as e:
        print(f"Error calculating BLEU: {e}")
        return 0.0


def extract_texts(response_text):
    """Extracts statistical and descriptive text using regex."""
    statistical_match = re.search(
        r"Factual statistical description:\n(.*?)\n\nDescriptive text:",
        response_text,
        re.DOTALL | re.IGNORECASE,
    )
    descriptive_match = re.search(
        r"Descriptive text:\n(.*)", response_text, re.DOTALL | re.IGNORECASE
    )

    statistical_text = statistical_match.group(1).strip() if statistical_match else ""
    descriptive_text = descriptive_match.group(1).strip() if descriptive_match else ""

    # Clean potential markdown code blocks
    statistical_text = re.sub(
        r"```(.*?)\n(.*?)```", r"\2", statistical_text, flags=re.DOTALL
    )
    descriptive_text = re.sub(
        r"```(.*?)\n(.*?)```", r"\2", descriptive_text, flags=re.DOTALL
    )

    statistical_text = statistical_text.strip()
    descriptive_text = descriptive_text.strip()

    return statistical_text, descriptive_text


def extract_textv2(response_text):

    # Clean potential markdown code blocks
    descriptive_text = re.sub(
        r"```(.*?)\n(.*?)```", r"\2", response_text, flags=re.DOTALL
    )
    descriptive_text = descriptive_text.strip()

    return descriptive_text


# --- Main Processing Logic ---

print("Starting script...")

# 1. Load JSON data
try:
    with open(JSON_FILE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Successfully loaded data from {JSON_FILE_PATH}")
except FileNotFoundError:
    print(f"ERROR: JSON file not found at {JSON_FILE_PATH}")
    exit()
except json.JSONDecodeError:
    print(f"ERROR: Could not decode JSON from {JSON_FILE_PATH}. Check file format.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred loading the JSON: {e}")
    exit()


all_rows = []
all_factors = set()
all_evaluations = set()
factor_predictions = {}  # To store factor_name -> pred_value mapping for each item

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

print("Processing JSON data...")
# 2. Iterate through JSON items
for item_id, item_data in data.items():
    print(f"  Processing item: {item_id}")
    if not isinstance(item_data, dict):
        print(
            f"    WARNING: Skipping item {item_id} due to unexpected data format (expected dict)."
        )
        continue

    response_text = item_data.get("wordalisation", "")
    full_description = item_data.get("description", "")

    if not response_text:
        print(
            f"    WARNING: No 'response' field found or empty for item {item_id}. Skipping."
        )
        continue

    # Extract statistical and descriptive text
    # _, descriptive_text = extract_texts(
    #     response_text
    # )  # We only need statistical_text for rows
    descriptive_text = extract_textv2(response_text)

    if not descriptive_text:
        print(
            f"    WARNING: Could not extract statistical description for item {item_id}."
        )
        # Decide if you want to skip or proceed with empty text
        # continue # Option to skip if no statistical text found

    # Extract factors and their predictions for this item
    item_factors = {}
    item_factor_names = []
    item_evaluations = set()
    for key, value in item_data.items():
        if key.endswith("_true"):
            factor_name = key[:-5]  # Remove '_true'
            item_factor_names.append(factor_name)
            all_factors.add(factor_name)
            # Store the prediction for this factor
            pred_key = f"{factor_name}_pred"
            if pred_key in item_data:
                pred_value = item_data[pred_key]
                item_factors[factor_name] = pred_value
                all_evaluations.add(pred_value)
                item_evaluations.add(pred_value)
            else:
                print(
                    f"    WARNING: Prediction key '{pred_key}' not found for factor '{factor_name}' in item {item_id}."
                )
                item_factors[factor_name] = None  # Handle missing prediction

    # Split statistical text into sentences
    sentences = split_into_sentences(descriptive_text)
    if not sentences and descriptive_text:  # If splitting failed but text exists
        print(
            f"    INFO: Sentence splitting yielded no results for non-empty text in {item_id}. Treating as one sentence."
        )
        sentences = [descriptive_text]
    elif not sentences:
        print(f"    INFO: No sentences found in statistical description for {item_id}.")

    # Process each sentence
    for i, sentence in enumerate(sentences):
        sentence_id = i + 1
        sentence_lower = sentence.lower()
        # replace "-" with " " to avoid tokenization issues
        sentence_lower = re.sub(r"-", "", sentence_lower)
        try:
            sentence_tokens = word_tokenize(sentence_lower)
        except Exception as e:
            print(
                f"    WARNING: Could not tokenize sentence {sentence_id} for {item_id}. Skipping BLEU matching. Error: {e}"
            )
            sentence_tokens = []

        matched_factors = []
        if (
            sentence_tokens and item_factor_names
        ):  # Only calculate BLEU if we have tokens and factors
            factor_scores = {}
            for factor in item_factor_names:
                factor_lower = factor.lower()
                # replace "-" with "" to avoid tokenization issues
                factor_lower = re.sub(r"-", "", factor_lower)
                try:
                    factor_tokens = word_tokenize(factor_lower)
                    # Remove stopwords and lemmatize if needed
                    factor_tokens = [
                        lemmatizer.lemmatize(w)
                        for w in factor_tokens
                        if w not in stop_words
                    ]
                    # do the same for sentence_tokens
                    sentence_tokens = [
                        lemmatizer.lemmatize(w)
                        for w in sentence_tokens
                        if w not in stop_words
                    ]
                    if (
                        factor_tokens
                    ):  # Ensure factor name is not empty after tokenization
                        bleu_score = calculate_bleu(factor_tokens, sentence_tokens)
                        factor_scores[factor] = bleu_score
                        # print(f"      Sentence {sentence_id}, Factor '{factor}', BLEU: {bleu_score:.4f}") # Debugging
                    else:
                        factor_scores[factor] = 0.0
                except Exception as e:
                    print(
                        f"    WARNING: Error processing factor '{factor}' for BLEU. Setting score to 0. Error: {e}"
                    )
                    factor_scores[factor] = 0.0

            # Find factors meeting the threshold
            high_score_factors = [
                f for f, score in factor_scores.items() if score >= BLEU_SCORE_THRESHOLD
            ]

            if high_score_factors:
                matched_factors.extend(high_score_factors)
                # print(f"      Sentence {sentence_id}: Matched factors above threshold ({BLEU_SCORE_THRESHOLD}): {high_score_factors}") # Debugging
            else:
                # print(f"      Sentence {sentence_id}: No factors met BLEU threshold {BLEU_SCORE_THRESHOLD}.") # Debugging
                pass  # Will default to "None" later

        # Create row(s) for the sentence
        if matched_factors:
            for factor_name in matched_factors:
                pred_value = item_factors.get(
                    factor_name
                )  # Get prediction for this specific factor
                if pred_value is None:
                    print(
                        f"    WARNING: Matched factor '{factor_name}' has no prediction value in item {item_id}. Setting Evaluation to None."
                    )
                    eval_value = "None"
                else:
                    eval_value = pred_value

                # d is the part of full_description matching | factor_name | ... | ... |
                # We can use regex to extract the relevant part of the description, not case sensitive
                d = re.search(
                    rf"\| {re.escape(factor_name)} \|.*?\| .*?\|",
                    full_description,
                    re.IGNORECASE,
                )
                if d:
                    d = d.group(0).strip()
                else:
                    d = ""
                all_rows.append(
                    {
                        "id": item_id,
                        "sentenceID": sentence_id,
                        "sentence_text": sentence,  # Add sentence text for context if needed (optional)
                        "factor": factor_name,
                        "true_value": item_data.get(f"{factor_name}_true", "None"),
                        "evaluation": eval_value,
                        "full_description": d,
                    }
                )
        else:
            # No factor matched well enough, create a row with "None"
            all_rows.append(
                {
                    "id": item_id,
                    "sentenceID": sentence_id,
                    "sentence_text": sentence,  # Optional
                    "factor": "None",
                    "true_value": "None",  # As per requirement
                    "evaluation": "None",  # As per requirement
                    "full_description": full_description.split(
                        "|:------|------:|:--------------------|\n"
                    )[1],
                }
            )

print(f"Processed {len(data)} items, generated {len(all_rows)} rows.")

# 3. Prepare data for Google Sheets
if not all_rows:
    print("No rows generated. Exiting.")
    exit()

df_all = pd.DataFrame(all_rows)
# Reorder columns as requested (optional sentence_text can be kept or removed)
# df = df[["id", "sentenceID", "factor", "evaluation"]]  # , "sentence_text"

# Prepare dropdown lists - ensure 'None' is included
factor_dropdown_list = sorted(list(all_factors)) + ["None"]
evaluation_dropdown_list = sorted(list(all_evaluations)) + ["None"]

# Ensure no duplicates in dropdown lists (though set conversion should handle this)
factor_dropdown_list = sorted(list(set(factor_dropdown_list)))
evaluation_dropdown_list = sorted(list(set(evaluation_dropdown_list)))


# 4. Authenticate and Connect to Google Sheets
print("Authenticating with Google Sheets...")
try:
    gc = authenticate_gsheet()
except Exception as e:
    print(f"ERROR: Failed to authenticate or connect to Google Sheets: {e}")
    print(
        "Please ensure 'credentials.json' is valid and you have authorized the application."
    )
    exit()


# 5. Create or Open Spreadsheet and Worksheet
print(f"Creating/Opening spreadsheet '{SPREADSHEET_NAME}'...")
try:
    spreadsheet = gc.create(SPREADSHEET_NAME)
    # Share with yourself if needed (optional, usually owner has access)
    # spreadsheet.share('your_email@example.com', perm_type='user', role='writer')
    print(f"Created new spreadsheet. URL: {spreadsheet.url}")
except gspread.exceptions.APIError as e:
    if "already exists" in str(e):
        print(f"Spreadsheet '{SPREADSHEET_NAME}' already exists. Opening it.")
        spreadsheet = gc.open(SPREADSHEET_NAME)
    else:
        print(f"ERROR: An API error occurred: {e}")
        exit()
except Exception as e:
    print(f"ERROR: Could not create or open spreadsheet '{SPREADSHEET_NAME}': {e}")
    exit()


ids = df_all["id"].unique()
# shuffle the ids
ids = list(ids)
import random

random.shuffle(ids)

ids = (
    [x for x in ids if "control" in x][:12]
    + [x for x in ids if "textual" in x][:12]
    + [x for x in ids if "numerical" in x][:12]
)

random.shuffle(ids)
# split ids in 4 groups
ids = [ids[i : i + len(ids) // 4] for i in range(0, len(ids), len(ids) // 4)]

for ind, id_list in enumerate(ids):
    # filter df_all by id_list
    df = df_all[df_all["id"].isin(id_list)]
    # Get or create worksheet
    WORKSHEET_NAME = f"{ttype}_{ind + 1}"
    try:
        worksheet = spreadsheet.worksheet(WORKSHEET_NAME)
        print(f"Using existing worksheet '{WORKSHEET_NAME}'. Clearing previous data...")
        worksheet.clear()  # Clear existing data before writing new data
    except gspread.exceptions.WorksheetNotFound:
        print(f"Worksheet '{WORKSHEET_NAME}' not found. Creating it.")
        worksheet = spreadsheet.add_worksheet(
            title=WORKSHEET_NAME, rows="1", cols="1"
        )  # Start small
    except Exception as e:
        print(f"Error accessing worksheet: {e}")
        exit()

    # 6. Write DataFrame to Worksheet
    print("Writing data to worksheet...")
    # Adjust worksheet size before writing might be necessary for large dataframes
    # worksheet.resize(rows=len(df) + 1, cols=len(df.columns)) # +1 for header row
    worksheet.update(
        [df.columns.values.tolist()] + df.values.tolist(),
        value_input_option="USER_ENTERED",
    )
    print("Data written successfully.")

    # make sentence_text column wrap
    worksheet.format("C2:C", {"wrapStrategy": "WRAP"})
    # make full_description column wrap
    worksheet.format("G2:G", {"wrapStrategy": "WRAP"})

    # 7. Set up Data Validation (Dropdowns)
    print("Setting up data validation (dropdowns)...")
    try:
        # Data validation for 'factor' column (Column C)
        factor_col_letter = gspread.utils.rowcol_to_a1(
            1, df.columns.get_loc("factor") + 1
        )[0]
        factor_range = (
            f"{factor_col_letter}2:{factor_col_letter}{len(df) + 1}"  # Start from row 2
        )
        # worksheet.data_validation(
        #     factor_range,
        #     condition_type="ONE_OF_LIST",
        #     condition_values=factor_dropdown_list,
        #     input_message="Select a factor or None.",
        #     strict=True,
        # )  # Set strict=True to only allow list values
        worksheet.add_validation(
            factor_range,
            condition_type=gspread.utils.ValidationConditionType.one_of_list,
            values=factor_dropdown_list,
            inputMessage="Select a factor or None.",
            strict=True,
            showCustomUi=True,
        )

        # Data validation for 'evaluation' column (Column D)
        eval_col_letter = gspread.utils.rowcol_to_a1(
            1, df.columns.get_loc("evaluation") + 1
        )[0]
        eval_range = (
            f"{eval_col_letter}2:{eval_col_letter}{len(df) + 1}"  # Start from row 2
        )
        # worksheet.data_validation(
        #     eval_range,
        #     condition_type="ONE_OF_LIST",
        #     condition_values=evaluation_dropdown_list,
        #     input_message="Select an evaluation status or None.",
        #     strict=True,
        # )
        worksheet.add_validation(
            eval_range,
            condition_type=gspread.utils.ValidationConditionType.one_of_list,
            values=evaluation_dropdown_list,
            inputMessage="Select a label or None.",
            strict=True,
            showCustomUi=True,
        )

        print("Data validation set successfully.")

    except Exception as e:
        print(f"ERROR: Could not set data validation: {e}")
        print("The sheet has been created/updated, but dropdowns might be missing.")

print("-" * 20)
print("Script finished successfully!")
print(f"Spreadsheet URL: {spreadsheet.url}")
print("-" * 20)
