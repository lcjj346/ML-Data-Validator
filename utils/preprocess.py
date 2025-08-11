# import re
# import pandas as pd

# def clean_phone_number(phone):
#     if pd.isnull(phone):
#         return ""
#     return re.sub(r"[^\d]", "", str(phone))

# def is_valid_format(phone):
#     """Assumes SG local numbers or general format for now"""
#     cleaned = clean_phone_number(phone)
#     return len(cleaned) >= 8 and len(cleaned) <= 15
