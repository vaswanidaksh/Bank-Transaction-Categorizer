"""
Text Preprocessor for Bank Transaction Narratives
-------------------------------------------------
Cleans, normalizes, and extracts meaningful merchant and semantic tokens
from messy bank transaction descriptions (UPI, NEFT, IMPS, POS, ACH, Card).
"""

import re
import string
from typing import List, Union
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Common banking noise words and protocol tokens to remove
BANK_STOPWORDS = {
    "upi", "cr", "dr", "neft", "imps", "rtgs", "pos", "ach", "nach", "cms",
    "txn", "txnid", "ref", "refno", "rrn", "at", "to", "by", "for", "in",
    "on", "of", "and", "the", "pay", "payment", "paid", "sent", "received",
    "swipe", "card", "ecom", "online", "transfer", "mand", "mandate",
    "wdl", "atm", "charges", "chg", "fee", "fees", "gst", "sms", "alert",
    "ind", "inr", "p2p", "p2m", "bill", "billpay"
}

# Bank names and handles to suppress so they don't override the merchant
BANK_IDENTIFIERS = {
    "hdfc", "icici", "sbi", "sbin", "axis", "utib", "kotak", "kkbk",
    "yesb", "punb", "barb", "paytm", "okaxis", "okhdfcbank", "oksbi",
    "okicici", "apl", "ybl", "ibl", "axl"
}

# Regex patterns
RE_CARD_MASK = re.compile(r"\b\d{4}[X*x]+\d{2,4}\b|\b[X*x]{4,}\b")
RE_REF_NUMBERS = re.compile(r"\b\d{6,}\b")
RE_IFSC = re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b", re.IGNORECASE)
RE_PUNCTUATION = re.compile(f"[{re.escape(string.punctuation)}/]")
RE_EXTRA_SPACES = re.compile(r"\s+")


def clean_description(text: str) -> str:
    """
    Cleans a single bank transaction description string.

    Pipeline:
    1. Lowercase
    2. Remove card masks (e.g. 4421XXXX1234)
    3. Remove IFSC codes (e.g. HDFC0001234)
    4. Remove long reference / transaction numbers (e.g. 432901238910)
    5. Replace punctuation, slashes, dashes with spaces
    6. Filter out banking noise keywords and bank abbreviations
    7. Remove standalone numbers and single characters
    8. Collapse extra whitespace
    """
    if not isinstance(text, str):
        return ""

    s = text.lower()

    # Step 1: Remove masked card numbers
    s = RE_CARD_MASK.sub(" ", s)

    # Step 2: Remove IFSC branch identifiers
    s = RE_IFSC.sub(" ", s)

    # Step 3: Remove long numeric reference IDs
    s = RE_REF_NUMBERS.sub(" ", s)

    # Step 4: Convert slashes and punctuation to whitespace
    s = RE_PUNCTUATION.sub(" ", s)

    # Step 5: Split into tokens and filter stopwords/bank codes
    tokens = s.split()
    cleaned_tokens = []
    for token in tokens:
        # Strip digits inside tokens if token is mixed like ref1234
        t = re.sub(r"\d+", "", token).strip()
        if len(t) <= 1:
            continue
        if t in BANK_STOPWORDS or t in BANK_IDENTIFIERS:
            continue
        cleaned_tokens.append(t)

    result = " ".join(cleaned_tokens)
    return RE_EXTRA_SPACES.sub(" ", result).strip()


class TransactionTextCleaner(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer for integration directly
    into Pipeline objects.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X: Union[List[str], pd.Series]) -> List[str]:
        if isinstance(X, pd.Series):
            return X.fillna("").apply(clean_description).tolist()
        return [clean_description(str(item)) for item in X]


if __name__ == "__main__":
    test_cases = [
        "UPI/CR/432423409123/SWIGGY/ICICI/payment",
        "POS 4421XXXX1234 DMART BANNERGHATTA BLR",
        "NEFT-CR-N12345678-OWNER RENT-HDFC0001234",
        "ACH D- NETFLIX ENTERTAINMENT SVCS MAND987123",
        "E-COM PURCHASE 5123XXXXXXXX9912 AMAZON INDIA IND",
        "UPI/DR/998811223344/rohit@okaxis/pay",
        "ATM CASH WDL BOM ATM402"
    ]
    print("Preprocessor Demo:\n" + "=" * 40)
    for raw in test_cases:
        cleaned = clean_description(raw)
        print(f"RAW:     {raw}")
        print(f"CLEANED: {cleaned}\n")

