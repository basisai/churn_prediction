"""
Constants.
"""
AREA_CODES = [408, 415, 510]

STATES = [
    "AK",
    "AL",
    "AR",
    "AZ",
    "CA",
    "CO",
    "CT",
    "DC",
    "DE",
    "FL",
    "GA",
    "HI",
    "IA",
    "ID",
    "IL",
    "IN",
    "KS",
    "KY",
    "LA",
    "MA",
    "MD",
    "ME",
    "MI",
    "MN",
    "MO",
    "MS",
    "MT",
    "NC",
    "ND",
    "NE",
    "NH",
    "NJ",
    "NM",
    "NV",
    "NY",
    "OH",
    "OK",
    "OR",
    "PA",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "UT",
    "VA",
    "VT",
    "WA",
    "WI",
    "WV",
    "WY",
]

SUBSCRIBER_FEATURES = [
    "Intl_Plan",
    "VMail_Plan",
    "VMail_Message",
    "CustServ_Calls",
    "Day_Mins",
    "Day_Calls",
    "Eve_Mins",
    "Eve_Calls",
    "Night_Mins",
    "Night_Calls",
    "Intl_Mins",
    "Intl_Calls",
]

FEATURE_COLS = SUBSCRIBER_FEATURES + \
    [f"Area_Code={area_code}" for area_code in AREA_CODES] + \
    [f"State={state}" for state in STATES]

TARGET_COL = "Churn"

USER_COL = "User_id"
