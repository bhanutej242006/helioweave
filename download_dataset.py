"""Re-download the requested-schema CSV, or import a manually downloaded Kaggle ZIP."""
import argparse,io,zipfile,urllib.request
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--zip',type=Path);args=p.parse_args()
url='https://www.kaggle.com/api/v1/datasets/download/stucom/solar-energy-power-generation-dataset'
try:
    content=args.zip.read_bytes() if args.zip else urllib.request.urlopen(url,timeout=120).read()
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        if 'spg.csv' not in z.namelist():raise ValueError('spg.csv is not present in this archive')
        out=Path(__file__).parent/'data/spg.csv';out.write_bytes(z.read('spg.csv'))
    print('Saved',out,'— run python train.py and restart Flask to use changes.')
except Exception as e:
    raise SystemExit(f'Download failed: {e}\nDownload from Kaggle and run python download_dataset.py --zip PATH_TO_ZIP')
