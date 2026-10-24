"""Download original Kaggle CSVs, or import an already downloaded ZIP."""
import argparse, io, zipfile, urllib.request
from pathlib import Path
URL='https://www.kaggle.com/api/v1/datasets/download/anikannal/solar-power-generation-data'
p=argparse.ArgumentParser(); p.add_argument('--zip',type=Path); args=p.parse_args()
try:
    content=args.zip.read_bytes() if args.zip else urllib.request.urlopen(URL,timeout=120).read()
    target=Path(__file__).parent/'data/raw'; target.mkdir(parents=True,exist_ok=True)
    expected={f'Plant_{n}_{kind}_Data.csv' for n in (1,2) for kind in ('Generation','Weather_Sensor')}
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        if not expected.issubset(z.namelist()): raise ValueError('Expected four Kaggle CSV files are missing')
        for name in sorted(expected): (target/name).write_bytes(z.read(name))
    print('Imported four original Kaggle files into',target)
except Exception as e:
    raise SystemExit(f'Download failed: {e}\nDownload the ZIP at https://www.kaggle.com/datasets/anikannal/solar-power-generation-data and run python download_dataset.py --zip YOUR_ZIP_PATH')
