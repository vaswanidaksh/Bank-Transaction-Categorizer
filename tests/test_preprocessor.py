import pytest
from src.preprocessor import clean_description, TransactionTextCleaner

def test_clean_upi_standard():
    raw = "UPI/CR/432423409123/SWIGGY/ICICI/payment"
    assert clean_description(raw) == "swiggy"

def test_clean_pos_with_card_mask():
    raw = "POS 4421XXXX1234 DMART BANNERGHATTA BLR"
    cleaned = clean_description(raw)
    assert "4421" not in cleaned
    assert "xxxx" not in cleaned
    assert "dmart" in cleaned

def test_clean_neft_rent():
    raw = "NEFT-CR-N12345678-OWNER RENT-HDFC0001234"
    cleaned = clean_description(raw)
    assert "owner rent" in cleaned
    assert "hdfc" not in cleaned
    assert "n12345678" not in cleaned

def test_clean_ach_netflix():
    raw = "ACH D- NETFLIX ENTERTAINMENT SVCS MAND987123"
    cleaned = clean_description(raw)
    assert "netflix" in cleaned

def test_sklearn_transformer():
    transformer = TransactionTextCleaner()
    input_data = [
        "UPI/CR/111111/ZOMATO/SBIN/pay",
        "E-COM PURCHASE 4111XXXXXXXX1111 FLIPKART"
    ]
    output = transformer.transform(input_data)
    assert len(output) == 2
    assert "zomato" in output[0]
    assert "flipkart" in output[1]

