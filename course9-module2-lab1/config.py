import sys
from pathlib import Path

PORT = 8000
HTTP_URL = f"http://127.0.0.1:{PORT}/mcp"

# Directory the document resource reads from, relative to the lab folder.
DOCUMENTS_DIR = Path("path")

SAMPLE_FILE_URLS = {
    "examples.txt": "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/aNE__JjH4DLNEibuNpfDlg/examples.txt",
    "README.txt": "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/tfoeGPInNoajVS0DSohdVg/README.txt",
}

# The stdio server is launched as a subprocess from the lab folder, with the
# interpreter that is running this code (the venv's, not whatever is on PATH).
SERVER_COMMAND = sys.executable
SERVER_ARGS = ["run_server.py"]
