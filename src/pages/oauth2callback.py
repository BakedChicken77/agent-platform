# pages/oauth2callback.py

import streamlit as st
import msal
import os
import logging

st.set_page_config(page_title="OAuth2 Callback", layout="centered")

# Enable MSAL debug logs if needed
# logging.basicConfig(level=logging.DEBUG)

# Load environment variables
CLIENT_ID = os.getenv("AZURE_AD_CLIENT_ID")
CLIENT_SECRET = os.getenv("AZURE_AD_CLIENT_SECRET")
TENANT_ID = os.getenv("AZURE_AD_TENANT_ID")
REDIRECT_URI = os.getenv("STREAMLIT_REDIRECT_URI")
API_CLIENT_ID = os.getenv("AZURE_AD_API_CLIENT_ID")
SCOPE = [f"api://{API_CLIENT_ID}/access_as_user"]

# Azure AD authority
AUTHORITY = f"https://login.microsoftonline.us/{TENANT_ID}"

# Initialize MSAL client
msal_app = msal.ConfidentialClientApplication(
    client_id=CLIENT_ID,
    authority=AUTHORITY,
    client_credential=CLIENT_SECRET,
)

# Extract query parameters
params = st.query_params
raw_code = params.get("code")

# Normalize the `code` to a usable string
if isinstance(raw_code, str):
    auth_code = raw_code
elif isinstance(raw_code, list):
    auth_code = raw_code[0]
else:
    auth_code = None

# Only proceed if a valid code is present and not already authenticated
if auth_code and "access_token" not in st.session_state:
    try:
        token_response = msal_app.acquire_token_by_authorization_code(
            auth_code,
            scopes=SCOPE,
            redirect_uri=REDIRECT_URI,
        )

        if "access_token" not in token_response:
            st.error("Failed to acquire access token.")
            st.json(token_response)  # show detailed error for debugging
            st.stop()

        # Success
        access_token = token_response["access_token"]
        st.session_state["access_token"] = access_token

        # Optional: save ID token, refresh token, user info if needed
        # st.session_state["id_token"] = token_response.get("id_token")
        # st.session_state["user_info"] = token_response.get("id_token_claims")

        # Clear the query params and redirect to main app
        st.query_params.clear()
        st.switch_page("streamlit_app.py")  # Requires Streamlit >= 1.32

    except Exception as e:
        st.error(f"OAuth callback error: {e}")
        st.stop()

elif not auth_code:
    st.warning("No authorization code found in query parameters.")
else:
    st.info("You are already signed in.")
