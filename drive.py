"""
Google Drive helpers: authenticate and list/download files in a folder.

First run: opens browser for OAuth2 consent, saves token to token.json.
Subsequent runs: uses cached token (auto-refreshed).

Setup (one-time):
  1. Go to https://console.cloud.google.com
  2. Create a project → Enable "Google Drive API"
  3. Create credentials → OAuth client ID → Desktop app → Download JSON
  4. Save as credentials.json in this directory
"""

import os
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
CREDENTIALS_FILE = "credentials.json"
TOKEN_FILE = "token.json"


def get_service():
    """Return an authenticated Google Drive API service."""
    creds = None
    if Path(TOKEN_FILE).exists():
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not Path(CREDENTIALS_FILE).exists():
                raise FileNotFoundError(
                    f"{CREDENTIALS_FILE} not found.\n"
                    "  1. Go to https://console.cloud.google.com\n"
                    "  2. Enable Google Drive API\n"
                    "  3. Create OAuth credentials (Desktop app)\n"
                    f"  4. Download and save as {CREDENTIALS_FILE}"
                )
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=8080)
        Path(TOKEN_FILE).write_text(creds.to_json())
    return build("drive", "v3", credentials=creds)


def list_folder_files(folder_id: str) -> dict[str, str]:
    """Return mapping of filename → webViewLink for all files in a Drive folder."""
    service = get_service()
    result: dict[str, str] = {}
    page_token = None

    while True:
        resp = (
            service.files()
            .list(
                q=f"'{folder_id}' in parents and trashed = false",
                fields="nextPageToken, files(name, id, webViewLink)",
                pageSize=200,
                pageToken=page_token,
            )
            .execute()
        )
        for f in resp.get("files", []):
            result[f["name"]] = f.get(
                "webViewLink", f"https://drive.google.com/file/d/{f['id']}/view"
            )
        page_token = resp.get("nextPageToken")
        if not page_token:
            break

    return result


def list_folder_files_with_ids(folder_id: str) -> list[dict]:
    """Return list of {name, id, webViewLink} for all files in a Drive folder."""
    service = get_service()
    results = []
    page_token = None

    while True:
        resp = (
            service.files()
            .list(
                q=f"'{folder_id}' in parents and trashed = false",
                fields="nextPageToken, files(id, name, mimeType, webViewLink)",
                pageSize=200,
                pageToken=page_token,
            )
            .execute()
        )
        results.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break

    return results


def extract_folder_id(folder_url_or_id: str) -> str:
    """Accept full Drive URL or bare folder ID."""
    if "folders/" in folder_url_or_id:
        return folder_url_or_id.split("folders/")[-1].split("?")[0]
    return folder_url_or_id
