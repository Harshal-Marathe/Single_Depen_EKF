"""
Session-state initialisation for the Rainbrain 2 app.
"""

import streamlit as st

_defaults = {
    "df": None,
    "prophet_results": None,
    "prophet_cols_added": [],   # list of "prophet_*" column names merged into df
    "config": None,
    "model_results": None,
    "model_fitted": False,
}


def init_session_state():
    for k, v in _defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
