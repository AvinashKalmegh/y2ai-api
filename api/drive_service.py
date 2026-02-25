import json
import os
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io


SCOPES = ['https://www.googleapis.com/auth/drive']

def get_drive_service():
    info = json.loads(os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"])

    credentials = service_account.Credentials.from_service_account_info(
        info,
        scopes=SCOPES
    )

    return build('drive', 'v3', credentials=credentials)


def get_latest_html(main_folder_id):
    drive_service = get_drive_service()

    results = drive_service.files().list(
        q=f"'{main_folder_id}' in parents and name='snapshot.html'",
        fields="files(id, name)"
    ).execute()

    files = results.get("files", [])
    if not files:
        return None

    file_id = files[0]['id']
    request = drive_service.files().get_media(fileId=file_id)
    return request.execute().decode("utf-8")
    

