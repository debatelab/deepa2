# config.py

from pathlib import Path 

package_dir = Path(__file__).parent.resolve()
template_dir:Path = Path(package_dir,'..','templates').resolve()
data_dir:Path = Path(package_dir,'..','data').resolve()

moral_maze_config = {
    'cache_dir': data_dir / "raw" / "aifdb" / "moral-maze",
    'templates_sp_ca': [
        'aifdb/source_paraphrase_ca-01.txt',
        'aifdb/source_paraphrase_ca-02.txt',
        'aifdb/source_paraphrase_ca-03.txt',
    ],
    'templates_sp_ra': [
        'aifdb/source_paraphrase_ra.txt',
    ],
    'corpora':[
        'http://corpora.aifdb.org/zip/britishempire'    
    ],
    'splits':  {"train":.8,"validation":.1,"test":.1},
}