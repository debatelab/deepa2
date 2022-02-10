"""Downloadmanager and other utilities"""

import requests


class DownloadManager:  # pylint: disable=too-few-public-methods
    """DownloadManager"""

    _CHUNK_SIZE = 32768
    _GDRIVE_URL = "https://docs.google.com/uc?export=download"

    @staticmethod
    def download_file_from_google_drive(file_id, destination):
        """download file from GDrive"""

        session = requests.Session()

        response = session.get(
            DownloadManager._GDRIVE_URL, params={"id": file_id}, stream=True
        )
        token = DownloadManager._get_confirm_token(response)

        if token:
            params = {"id": file_id, "confirm": token}
            response = session.get(
                DownloadManager._GDRIVE_URL, params=params, stream=True
            )

        DownloadManager._save_response_content(response, destination)

    @staticmethod
    def _get_confirm_token(response):
        """gdrive token"""
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                return value

        return None

    @staticmethod
    def _save_response_content(response, destination):
        """save response"""

        with open(destination, "wb") as file:
            for chunk in response.iter_content(DownloadManager._CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    file.write(chunk)
