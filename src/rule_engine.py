"""
Deterministic & Fuzzy Rule Engine for Bank Transactions
--------------------------------------------------------
Provides rapid, high-confidence keyword and pattern lookups for standard merchants,
salary keywords, tax refunds, and recurring bill signatures.
"""

from typing import Optional, Tuple, Dict
import re

# High-confidence exact keywords/tokens mapped to categories
RULE_MAP = {
    # Food & Dining
    "Food & Dining": [
        "swiggy", "zomato", "dominos", "mcdonalds", "starbucks", "kfc",
        "burger king", "chaipoint", "haldirams", "biryani", "subway",
        "pizza hut", "barbeque nation", "cafe coffee day", "third wave coffee",
        "social offline", "behrouz", "faasos", "wow momo"
    ],
    # Groceries
    "Groceries": [
        "blinkit", "zepto", "bigbasket", "dmart", "instamart", "natures basket",
        "reliance smart", "spencers retail", "supermarket", "star bazaar",
        "kirana", "bb daily", "country delight", "licious", "freshtohome"
    ],
    # Rent & Housing
    "Rent & Housing": [
        "nobroker rent", "housing com rent", "maintenance soc", "mygate",
        "apartment association", "owner rent", "urban company", "house maid",
        "flat rent", "landlord rent", "society maintenance"
    ],
    # Transport
    "Transport": [
        "uber", "ola cabs", "rapido", "fastag", "namma metro", "delhi metro",
        "dmrc", "irctc", "makemytrip", "petrol pump", "fuel station", "iocl",
        "hpcl", "bpcl", "shell auto", "red bus", "indigo airlines", "flight"
    ],
    # Shopping
    "Shopping": [
        "amazon", "flipkart", "myntra", "ajio", "zara", "h and m", "nykaa",
        "tata cliq", "decathlon", "croma", "reliance digital", "meesho",
        "uniqlo", "lifestyle stores"
    ],
    # Health & Medical
    "Health & Medical": [
        "apollo pharmacy", "tata 1mg", "pharmeasy", "lal pathlabs", "max healthcare",
        "fortis hospital", "cult fit", "medplus", "metropolis labs", "practo",
        "diagnostics", "dental clinic", "pharmacy", "hospital", "clinic"
    ],
    # Bills & Utilities
    "Bills & Utilities": [
        "airtel", "jio recharge", "bescom", "tata power", "adani electricity",
        "mahanagar gas", "act fibernet", "broadband", "hathway", "water supply",
        "indraprastha gas", "igl", "electricity bill", "utility"
    ],
    # Entertainment
    "Entertainment": [
        "netflix", "spotify", "prime video", "hotstar", "bookmyshow",
        "pvr cinemas", "inox", "steam games", "youtube premium", "sonyliv",
        "playstation network", "disney"
    ],
    # Education
    "Education": [
        "coursera", "udemy", "scaler academy", "upgrad", "tuition",
        "exam fee", "duolingo", "khan academy", "british council", "edx"
    ],
    # EMI & Loans
    "EMI & Loans": [
        "home loan emi", "auto loan emi", "bajaj finserv", "cred cc bill",
        "sbi cards", "slice super", "personal loan", "loan emi", "minimum due",
        "tata capital", "credit card payment"
    ],
    # Income
    "Income": [
        "salary credit", "payroll credit", "salary neft", "freelance invoice",
        "dividend credit", "income tax refund", "bonus payout", "interest earned"
    ],
    # Other / Uncategorized
    "Other / Uncategorized": [
        "atm cash wdl", "amc charges", "forex markup", "sms alert charges",
        "min bal charges", "card fee"
    ]
}


class RuleEngine:
    """Keyword-driven classification engine for sub-millisecond lookups."""

    def __init__(self):
        # Compile category regexes for high-performance scanning
        self.compiled_rules = {}
        for category, patterns in RULE_MAP.items():
            # Match boundary or substring
            regex_str = r"\b(" + "|".join(re.escape(p) for p in patterns) + r")\b"
            self.compiled_rules[category] = re.compile(regex_str, re.IGNORECASE)

    def match(self, cleaned_text: str) -> Optional[Tuple[str, float]]:
        """
        Attempts to match clean text against curated merchant/category patterns.
        Returns (category, confidence=1.0) if found, else None.
        """
        if not cleaned_text:
            return None

        # Check Income first because salary markers are explicit
        for category in ["Income", "EMI & Loans"]:
            if self.compiled_rules[category].search(cleaned_text):
                return category, 1.0

        for category, regex in self.compiled_rules.items():
            if regex.search(cleaned_text):
                return category, 0.99

        return None

