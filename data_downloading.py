import os
import gdown
import zipfile

def download_and_extract_data():
    """
    Downloads anime recommendation data from Google Drive and extracts to dat/ folder
    """
    print("Creating dat directory if it doesn't exist...")
    os.makedirs("dat", exist_ok=True)
    
    print("Downloading anime recommendation dataset from Google Drive...")
    
    # URL for the Anime Recommendations Database zip file from Google Drive
    # Replace this URL with your actual Google Drive file URL
    url = "https://drive.google.com/file/d/15k2324qvhaA_1XQR6txyqgbbFeKAPTA_/view?usp=drive_link"
    
    # Download destination
    zip_path = "dat/anime_data.zip"
    
    # Download the file
    gdown.download(url, zip_path, quiet=False)
    
    print(f"Dataset downloaded to {zip_path}")
    
    # Extract the files
    print("Extracting zip file...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("dat/")
    
    print("Cleaning up...")
    os.remove(zip_path)  # Remove the zip file after extraction
    
    print("Dataset ready in the dat/ directory")
    
    # Print list of files in the dat directory
    print("\nFiles in dat/ directory:")
    for file in os.listdir("dat/"):
        print(f"- {file}")

if __name__ == "__main__":
    download_and_extract_data()