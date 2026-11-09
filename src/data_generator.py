"""
High-Fidelity Indian & Global Bank Transaction Synthetic Data Generator
------------------------------------------------------------------------
Generates realistic bank statement records spanning 13 financial categories,
emulating authentic Indian protocols (UPI, NEFT, IMPS, POS, ACH, ATM) with
realistic merchant distribution, noise, card masks, branch codes, and reference IDs.
"""

import argparse
import random
from datetime import datetime, timedelta
import pandas as pd
from faker import Faker

fake = Faker("en_IN")

CATEGORIES = [
    "Food & Dining",
    "Groceries",
    "Rent & Housing",
    "Transport",
    "Shopping",
    "Health & Medical",
    "Bills & Utilities",
    "Entertainment",
    "Transfers",
    "Education",
    "EMI & Loans",
    "Income",
    "Other / Uncategorized",
]

# Merchant dictionaries mapped to standard category
MERCHANTS = {
    "Food & Dining": [
        "SWIGGY", "ZOMATO", "DOMINOS PIZZA", "MCDONALDS", "STARBUCKS", "KFC",
        "BURGER KING", "CHAIPOINT", "HALDIRAMS", "BIRYANI BY KILO", "SUBWAY",
        "PIZZA HUT", "BARBEQUE NATION", "CAFE COFFEE DAY", "THIRD WAVE COFFEE",
        "SOCIAL OFFLINE", "SWEET TRUTH", "BEHROUZ BIRYANI", "FAASOS", "WOW MOMO"
    ],
    "Groceries": [
        "BLINKIT", "ZEPTO", "BIGBASKET", "DMART", "SWIGGY INSTAMART", "NATURES BASKET",
        "RELIANCE SMART", "SPENCERS RETAIL", "MORE SUPERMARKET", "STAR BAZAAR",
        "LOCAL VEGETABLE VENDOR", "KIRANA STORE", "BB DAILY", "COUNTRY DELIGHT",
        "LILICIOUS MEAT", "FRESH TO HOME"
    ],
    "Rent & Housing": [
        "NOBROKER RENT", "HOUSING COM RENTPAY", "MAINTENANCE SOC", "MYGATE MAINTENANCE",
        "APARTMENT ASSOCIATION", "LANDLORD RENT TRANSFER", "URBAN COMPANY CLEANING",
        "HOUSE MAID WAGES", "SECURITY DEPOSIT FLAT", "URBAN COMPANY HOME SERVICES"
    ],
    "Transport": [
        "UBER INDIA", "OLA CABS", "RAPIDO BIKE", "FASTAG RECHARGE", "NAMMA METRO",
        "DELHI METRO DMRC", "IRCTC TICKETS", "MAKEMYTRIP FLIGHTS", "HPCL PETROL PUMP",
        "BPCL FUEL STATION", "IOCL PETROL", "SHELL AUTO CARE", "RED BUS", "INDIGO AIRLINES"
    ],
    "Shopping": [
        "AMAZON INDIA", "FLIPKART", "MYNTRA", "AJIO", "ZARA RETAIL", "H AND M",
        "NYKAA BEAUTY", "TATA CLIQ", "DECATHLON SPORTS", "CROMA ELECTRONICS",
        "RELIANCE DIGITAL", "MEESHO", "UNIQLO INDIA", "LIFESTYLE STORES"
    ],
    "Health & Medical": [
        "APOLLO PHARMACY", "TATA 1MG", "PHARMEASY", "DR LAL PATHLABS", "MAX HEALTHCARE",
        "FORTIS HOSPITAL", "CULT FIT GYM", "MEDPLUS PHARMACY", "METROPOLIS LABS",
        "PRACTO CONSULTATION", "HEALTHIANS DIAGNOSTICS", "DENTAL CLINIC"
    ],
    "Bills & Utilities": [
        "AIRTEL PREPAID", "JIO RECHARGE", "BESCOM ELECTRICITY", "TATA POWER",
        "ADANI ELECTRICITY", "MAHANAGAR GAS LIMITED", "ACT FIBERNET BROADBAND",
        "HATHWAY CABLE", "WATER SUPPLY BOARD", "INDRAPRASTHA GAS IGL", "VI POSTPAID"
    ],
    "Entertainment": [
        "NETFLIX INDIA", "SPOTIFY INDIA", "AMAZON PRIME VIDEO", "DISNEY HOTSTAR",
        "BOOKMYSHOW", "PVR CINEMAS", "INOX LEISURE", "STEAM GAMES", "YOUTUBE PREMIUM",
        "SONYLIV", "APPLE SERVICES", "PLAYSTATION NETWORK"
    ],
    "Transfers": [
        "UPI P2P TRANSFER", "SENT TO FRIEND", "SPLITWISE SETTLE", "IMPS TO SISTER",
        "NEFT TO FAMILY", "GOOGLE PAY PEER TRANSFER", "PHONEPE WALLET TOPUP",
        "PAYTM SEND MONEY", "FUNDS TRANSFER P2P", "CASH TRANSFER"
    ],
    "Education": [
        "COURSERA", "UDEMY ONLINE", "SCALER ACADEMY", "UPGRAD EDUCATION",
        "SCHOOL TUITION FEES", "COLLEGE SEMESTER EXAM FEE", "DUOLINGO SUBSCRIPTION",
        "KHAN ACADEMY DONATION", "BRITISH COUNCIL IELTS", "EDX ONLINE COURSES"
    ],
    "EMI & Loans": [
        "HDFC BANK HOME LOAN EMI", "ICICI AUTO LOAN EMI", "BAJAJ FINSERV EMI",
        "CRED CC BILL PAYMENT", "SBI CARDS PAYMENT", "SLICE SUPER CARD EMI",
        "KOTAK PERSONAL LOAN", "CREDIT CARD MINIMUM DUE", "TATA CAPITAL EMI"
    ],
    "Income": [
        "SALARY FOR THE MONTH", "TCS PAYROLL CREDIT", "INFOSYS SALARY NEFT",
        "ACCENTURE MONTHLY SALARY", "CLIENT FREELANCE INVOICE", "DIVIDEND CREDIT BSE",
        "INCOME TAX REFUND", "INTEREST EARNED SAVINGS", "BONUS PAYOUT CREDIT"
    ],
    "Other / Uncategorized": [
        "ATM CASH WITHDRAWAL", "ANNUAL CARD MAINTENANCE FEE", "FOREX MARKUP CHARGES",
        "CHEQUE BOOK CHARGES", "PENAL INTEREST", "MISCELLANEOUS DEBIT",
        "IMPS CHARGES", "SMS ALERT FEES", "SUSPENSE ACCOUNT REVERSAL"
    ]
}

INDIAN_BANKS = ["HDFC", "ICICI", "SBIN", "UTIB", "KKBK", "YESB", "PUNB", "BARB"]
CITIES = ["BLR", "BOM", "DEL", "HYD", "MAA", "PNQ", "CCU", "AMD"]


def _generate_upi_narrative(merchant: str) -> str:
    ref = random.randint(100000000000, 999999999999)
    bank = random.choice(INDIAN_BANKS)
    handle = random.choice(["okaxis", "okhdfcbank", "paytm", "apl", "ybl"])
    clean_m = merchant.lower().replace(" ", "")
    style = random.choice([1, 2, 3, 4])
    if style == 1:
        return f"UPI/CR/{ref}/{merchant.upper()}/{bank}/payment"
    elif style == 2:
        return f"UPI/DR/{ref}/{clean_m}@{handle}/{bank}/pay"
    elif style == 3:
        return f"UPI-{clean_m}-{ref}-{bank}"
    else:
        return f"UPI/{clean_m.upper()}@{handle}/REF{ref}/SENT"


def _generate_pos_narrative(merchant: str) -> str:
    card_mask = f"{random.randint(4000, 5999)}XXXXXXXX{random.randint(1000, 9999)}"
    city = random.choice(CITIES)
    style = random.choice([1, 2, 3])
    if style == 1:
        return f"POS {card_mask} {merchant.upper()} {city}"
    elif style == 2:
        return f"CARD SWIPE {card_mask} AT {merchant.upper()} IND"
    else:
        return f"E-COM PURCHASE {card_mask} {merchant.upper()}"


def _generate_neft_imps_narrative(merchant: str) -> str:
    ref = f"N{random.randint(10000000, 99999999)}"
    bank = random.choice(INDIAN_BANKS)
    branch = f"{bank}000{random.randint(1000, 9999)}"
    style = random.choice([1, 2])
    if style == 1:
        return f"NEFT-CR-{ref}-{merchant.upper()}-{branch}"
    else:
        return f"IMPS/{random.randint(1000000000, 9999999999)}/{merchant.upper()}/TXN"


def _generate_ach_narrative(merchant: str) -> str:
    mandate_id = f"MAND{random.randint(100000, 999999)}"
    return f"ACH D- {merchant.upper()} {mandate_id}"


def _generate_raw_description(category: str, merchant: str) -> str:
    """Generate realistic statement narration based on category and merchant."""
    if category == "Income":
        ref = random.randint(100000, 999999)
        templates = [
            f"CMS/SALARY/{merchant.upper()}/OCTOBER",
            f"NEFT-CR-SAL-{ref}-{merchant.upper()}",
            f"BY TRANSFER-SALARY CREDIT {merchant.upper()}",
            f"{merchant.upper()} SALARY NEFT REF{ref}",
            f"ACH C- {merchant.upper()} PAYROLL"
        ]
        return random.choice(templates)

    if category == "Transfers":
        friend_name = fake.first_name().upper()
        ref = random.randint(100000000000, 999999999999)
        templates = [
            f"UPI/CR/{ref}/PAY TO {friend_name}/HDFC",
            f"UPI/DR/{ref}/{friend_name.lower()}@okaxis",
            f"IMPS-P2P-{ref}-{friend_name}",
            f"NEFT-FAMILY TRANSFER TO {friend_name}",
            f"SPLITWISE SETTLEMENT {friend_name}"
        ]
        return random.choice(templates)

    if category == "Other / Uncategorized":
        templates = [
            f"ATM CASH WDL {random.choice(CITIES)} ATM{random.randint(100, 999)}",
            "ANNUAL DEBIT CARD AMC CHARGES PLUS GST",
            f"FOREX MARKUP TXN {random.randint(10000, 99999)}",
            "SMS ALERT CHARGES Q3",
            "CONSOLIDATED CHARGES DEBIT",
            "MIN BAL CHARGES NON MAINTENANCE"
        ]
        return random.choice(templates)

    # For other categories, assign realistic channels
    weights = [0.55, 0.25, 0.12, 0.08]  # UPI, POS, NEFT/IMPS, ACH
    channel = random.choices(["UPI", "POS", "NEFT_IMPS", "ACH"], weights=weights)[0]

    if channel == "UPI":
        return _generate_upi_narrative(merchant)
    elif channel == "POS":
        return _generate_pos_narrative(merchant)
    elif channel == "NEFT_IMPS":
        return _generate_neft_imps_narrative(merchant)
    else:
        return _generate_ach_narrative(merchant)


def _sample_amount(category: str) -> float:
    """Return realistic amounts (INR) per category."""
    ranges = {
        "Food & Dining": (80.0, 3500.0),
        "Groceries": (150.0, 7500.0),
        "Rent & Housing": (8000.0, 55000.0),
        "Transport": (40.0, 4500.0),
        "Shopping": (300.0, 25000.0),
        "Health & Medical": (100.0, 12000.0),
        "Bills & Utilities": (200.0, 8000.0),
        "Entertainment": (199.0, 2500.0),
        "Transfers": (100.0, 25000.0),
        "Education": (500.0, 45000.0),
        "EMI & Loans": (3000.0, 45000.0),
        "Income": (25000.0, 250000.0),
        "Other / Uncategorized": (15.0, 10000.0),
    }
    low, high = ranges.get(category, (50.0, 5000.0))
    return round(random.uniform(low, high), 2)


def generate_dataset(num_samples: int = 6000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic bank transactions dataframe."""
    random.seed(seed)
    Faker.seed(seed)

    records = []
    base_date = datetime(2026, 1, 1)

    for i in range(num_samples):
        txn_id = f"TXN_{2026000000 + i}"
        days_offset = random.randint(0, 300)
        txn_date = (base_date + timedelta(days=days_offset)).strftime("%Y-%m-%d")

        # Category selection with realistic distribution
        # Higher volume in food, groceries, transport, shopping
        cat_weights = [
            0.18,  # Food & Dining
            0.15,  # Groceries
            0.05,  # Rent & Housing
            0.14,  # Transport
            0.12,  # Shopping
            0.06,  # Health & Medical
            0.08,  # Bills & Utilities
            0.06,  # Entertainment
            0.07,  # Transfers
            0.03,  # Education
            0.03,  # EMI & Loans
            0.015, # Income
            0.015  # Other / Uncategorized
        ]
        category = random.choices(CATEGORIES, weights=cat_weights)[0]
        merchants_list = MERCHANTS[category]
        merchant = random.choice(merchants_list)

        raw_desc = _generate_raw_description(category, merchant)
        amount = _sample_amount(category)
        txn_type = "CREDIT" if category == "Income" else "DEBIT"

        records.append({
            "transaction_id": txn_id,
            "date": txn_date,
            "raw_description": raw_desc,
            "merchant_tag": merchant,
            "amount": amount,
            "transaction_type": txn_type,
            "category": category
        })

    df = pd.DataFrame(records)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic bank transactions.")
    parser.add_argument("--num-samples", type=int, default=6000, help="Number of records to generate")
    parser.add_argument("--output", type=str, default="data/raw/transactions.csv", help="Output CSV path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    df = generate_dataset(num_samples=args.num_samples, seed=args.seed)
    df.to_csv(args.output, index=False)
    print(f"Generated {len(df)} transactions -> {args.output}")
    print("\nCategory Distribution:")
    print(df["category"].value_counts())

