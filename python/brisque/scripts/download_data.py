import os

from google_drive_downloader import GoogleDriveDownloader as gdd

_CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


def main():
    file_id = "1AyEPQbMsjURQEdambu5TJW_ffklZOJu6"

    gdd.download_file_from_google_drive(
        file_id=file_id,
        dest_path=os.path.join(_CURRENT_DIR, "..", "data/Market3.hdr"),
        unzip=True,
    )


if __name__ == "__main__":
    main()
