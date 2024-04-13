import sys
import gdown

def download_from_google_drive(file_id, output_filename):
    url = 'https://drive.google.com/uc?id=' + file_id
    gdown.download(url, output_filename, quiet=False)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python download_from_google_drive.py <file_id> <output_filename>")
        sys.exit(1)

    file_id = sys.argv[1]
    output_filename = sys.argv[2]

    download_from_google_drive(file_id, output_filename)
