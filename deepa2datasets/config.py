# config.py

from pathlib import Path  # pathlib is seriously awesome!

template_dir = Path('../templates')
data_dir = Path('/path/to/some/logical/parent/dir')
data_path = data_dir / 'my_file.csv'  # use feather files if possible!!!