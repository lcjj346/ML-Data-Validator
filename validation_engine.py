import re

def validate_phone(phone: str):
    return phone.isdigit() and len(phone) >= 8

def validate_blood_sugar(value: float):
    return 70 <= value <= 200

def validate_report_id(report_id: str):
    return bool(re.match(r'^ccs\d{4}$', report_id))