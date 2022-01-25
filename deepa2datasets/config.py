# config.py

from pathlib import Path 

package_dir = Path(__file__).parent.resolve()
template_dir:Path = Path(package_dir,'..','templates').resolve()
data_dir:Path = Path(package_dir,'..','data').resolve()
