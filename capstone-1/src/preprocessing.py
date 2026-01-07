# src/preprocessing.py

import pandas as pd
import numpy as np

# -----------------------------
# Mapping dictionaries
# -----------------------------

marital_status_mapping = {
    1: 'single',
    2: 'married',
    3: 'widower',
    4: 'divorced',
    5: 'facto union',
    6: 'legally separated'
}

application_mode_mapping = {
    1: "1st phase - general contingent",
    2: "Ordinance No. 612/93",
    5: "1st phase - special contingent (Azores)",
    7: "Holders of other higher courses",
    10: "Ordinance No. 854-B/99",
    15: "International student (bachelor)",
    16: "1st phase - special contingent (Madeira)",
    17: "2nd phase - general contingent",
    18: "3rd phase - general contingent",
    26: "Ordinance No. 533-A/99 item b2) (Different Plan)",
    27: "Ordinance No. 533-A/99 item b3) (Other Institution)",
    39: "Over 23 years old",
    42: "Transfer",
    43: "Change of course",
    44: "Technological specialization diploma holders",
    51: "Change of institution/course",
    53: "Short cycle diploma holders",
    57: "Change of institution/course (International)"
}

attendance_mapping = {1: "daytime", 0: "evening"}

prev_qual_mapping = {
    1: "Secondary education",
    2: "Higher education - bachelor's degree",
    3: "Higher education - degree",
    4: "Higher education - master's",
    5: "Higher education - doctorate",
    6: "Frequency of higher education",
    9: "12th year - not completed",
    10: "11th year - not completed",
    12: "Other - 11th year",
    14: "10th year",
    15: "10th year - not completed",
    19: "Basic education 3rd cycle",
    38: "Basic education 2nd cycle",
    39: "Technological specialization course",
    40: "Higher education - degree",
    42: "Professional higher technical course",
    43: "Higher education - master’s"
}

nationality_mapping = {
    1: "Portuguese",
    2: "German",
    6: "Spanish",
    11: "Italian",
    13: "Dutch",
    14: "English",
    17: "Lithuanian",
    21: "Angolan",
    22: "Cape Verdean",
    24: "Guinean",
    25: "Mozambican",
    26: "Santomean",
    32: "Turkish",
    41: "Brazilian",
    62: "Romanian",
    100: "Moldova (Republic of)",
    101: "Mexican",
    103: "Ukrainian",
    105: "Russian",
    108: "Cuban",
    109: "Colombian"
}

binary_mapping = {0: "No", 1: "Yes"}

course_mapping = {
    33: "Biofuel Production Technologies",
    171: "Animation and Multimedia Design",
    8014: "Social Service (evening)",
    9003: "Agronomy",
    9070: "Communication Design",
    9085: "Veterinary Nursing",
    9119: "Informatics Engineering",
    9130: "Equinculture",
    9147: "Management",
    9238: "Social Service",
    9254: "Tourism",
    9500: "Nursing",
    9556: "Oral Hygiene",
    9670: "Advertising and Marketing Management",
    9773: "Journalism and Communication",
    9853: "Basic Education",
    9991: "Management (evening)"
}

gender_mapping = {1: "male", 0: "female"}

mother_occupation_mapping = {
    0: "Student",
    1: "Representatives of the Legislative Power and Executive Bodies",
    2: "Specialists in Intellectual and Scientific Activities",
    3: "Intermediate Level Technicians and Professions",
    4: "Administrative Staff",
    5: "Personal Services, Security and Safety Workers",
    6: "Farmers and Skilled Agricultural Workers",
    7: "Skilled Workers in Industry, Construction and Crafts",
    8: "Machine Operators and Assembly Workers",
    9: "Unskilled Workers",
    10: "Armed Forces",
    90: "Other",
    99: "Unknown"
}

father_occupation_mapping = mother_occupation_mapping.copy()
application_order_mapping = {i: f"{i+1}th choice" for i in range(9)}
fee_upto_date_mapping = {1: "Yes", 0: "No"}


# -----------------------------
# Core preprocessing
# -----------------------------

def apply_mappings(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df.rename(columns={
        "Nacionality": "Nationality",
        "Daytime/evening attendance\t": "Daytime/evening attendance"
    }, inplace=True)

    df["Marital status"] = df["Marital status"].map(marital_status_mapping)
    df["Application mode"] = df["Application mode"].map(application_mode_mapping)
    df["Course"] = df["Course"].map(course_mapping)
    df["Application order"] = df["Application order"].map(application_order_mapping)
    df["Gender"] = df["Gender"].map(gender_mapping)
    df["Daytime/evening attendance"] = df["Daytime/evening attendance"].map(attendance_mapping)
    df["Previous qualification"] = df["Previous qualification"].map(prev_qual_mapping)
    df["Nationality"] = df["Nationality"].map(nationality_mapping)
    df["Tuition fees up to date"] = df["Tuition fees up to date"].map(fee_upto_date_mapping)

    df["Mother's qualification"] = df["Mother's qualification"].map(prev_qual_mapping)
    df["Father's qualification"] = df["Father's qualification"].map(prev_qual_mapping)
    df["Mother's occupation"] = df["Mother's occupation"].map(mother_occupation_mapping)
    df["Father's occupation"] = df["Father's occupation"].map(father_occupation_mapping)

    for col in [
        "Displaced",
        "Educational special needs",
        "Debtor",
        "Scholarship holder",
        "International"
    ]:
        df[col] = df[col].map(binary_mapping)

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = apply_mappings(df)

    binary_cols = [
        "Displaced",
        "Debtor",
        "Scholarship holder",
        "International",
        "Educational special needs"
    ]

    for col in binary_cols:
        df[col] = df[col].map({"No": 0, "Yes": 1}).fillna(0).astype(int)

    df["financial_risk"] = (
        (df["Debtor"] == 1) |
        (df["Tuition fees up to date"] == "No")
    ).astype(int)

    df["tuition_not_up_to_date"] = (df["Tuition fees up to date"] == "No").astype(int)

    df["approval_rate_1st_sem"] = (
        df["Curricular units 1st sem (approved)"] /
        df["Curricular units 1st sem (enrolled)"].replace(0, np.nan)
    )

    df["approval_rate_2nd_sem"] = (
        df["Curricular units 2nd sem (approved)"] /
        df["Curricular units 2nd sem (enrolled)"].replace(0, np.nan)
    )

    df["approval_rate_change"] = (
        df["approval_rate_2nd_sem"] - df["approval_rate_1st_sem"]
    )

    df["total_enrolled_units"] = (
        df["Curricular units 1st sem (enrolled)"] +
        df["Curricular units 2nd sem (enrolled)"]()
    )