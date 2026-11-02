"""Download the public-domain King County housing CSV used by this group.

Run from the repository root: python -m src.download_kc
"""

from io import BytesIO
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

SOURCE_URL = (
    "https://raw.githubusercontent.com/jmatth11/"
    "King-County-House-Data-Set/master/kc_house_data.csv"
)
DESTINATION = Path("data/raw/kc_house_data.csv")


def download(destination: Path = DESTINATION) -> Path:
    request = Request(SOURCE_URL, headers={"User-Agent": "CS469-data-coursework"})
    with urlopen(request, timeout=60) as response:
        content = response.read()
    dataframe = pd.read_csv(BytesIO(content))
    essential = {"id", "date", "price", "sqft_living", "zipcode"}
    if len(dataframe) != 21613 or not essential.issubset(dataframe.columns):
        raise ValueError("Downloaded CSV did not match the expected King County dataset")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)
    print(f"Saved {len(dataframe):,} rows to {destination}")
    return destination


if __name__ == "__main__":
    download()
