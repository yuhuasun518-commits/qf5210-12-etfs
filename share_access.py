"""Require a bearer link for hosted viewing; local use needs an explicit flag."""
import os
from secrets import compare_digest

import streamlit as st


def token_matches(expected, supplied):
    return (isinstance(expected, str) and len(expected) >= 32
            and isinstance(supplied, str)
            and compare_digest(expected.encode("utf-8"), supplied.encode("utf-8")))


def require_shared_link():
    # This flag is server-side only. URL parameters cannot enable local mode.
    if os.environ.get("QF5210_LOCAL_MODE") == "1":
        return
    try:
        expected = st.secrets.get("share_token")
    except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
        expected = None
    if not isinstance(expected, str) or len(expected) < 32:
        st.title("Project dashboard")
        st.info("The owner is setting up sharing. Please use the shared project link when it is ready.")
        st.stop()
    if not token_matches(expected, st.query_params.get("access", "")):
        st.title("Project dashboard")
        st.info("Open the complete project link provided by the owner to view the dashboard and download its files.")
        st.stop()
