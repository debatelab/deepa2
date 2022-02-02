# config.py

from pathlib import Path 

package_dir = Path(__file__).parent.resolve()
template_dir:Path = Path(package_dir,'..','templates').resolve()
data_dir:Path = Path(package_dir,'..','data').resolve()

moral_maze_config = {
    'name':'moral-maze',
    'cache_dir': data_dir / "raw" / "aifdb" / "moral-maze",
    'templates_sp_ca': [
        'aifdb/source_paraphrase_ca-01.txt',
        'aifdb/source_paraphrase_ca-02.txt',
        'aifdb/source_paraphrase_ca-03.txt',
        'aifdb/source_paraphrase_ca-04.txt',
        'aifdb/source_paraphrase_ca-04.txt',
    ],
    'templates_sp_ra': [
        'aifdb/source_paraphrase_ra-01.txt',
        'aifdb/source_paraphrase_ra-02.txt',
    ],
    'corpora':[
        'http://corpora.aifdb.org/zip/britishempire',
        'http://corpora.aifdb.org/zip/Money',
        'http://corpora.aifdb.org/zip/welfare',
        'http://corpora.aifdb.org/zip/schemes',
        'http://corpora.aifdb.org/zip/problem',
        'http://corpora.aifdb.org/zip/mm2012',
        'http://corpora.aifdb.org/zip/mm2012a',
        'http://corpora.aifdb.org/zip/bankingsystem',
        'http://corpora.aifdb.org/zip/mm2012b',
        'http://corpora.aifdb.org/zip/mmbs2',
        'http://corpora.aifdb.org/zip/mm2012c',
        'http://corpora.aifdb.org/zip/MMSyr',
        'http://corpora.aifdb.org/zip/MoralMazeGreenBelt',
        'http://corpora.aifdb.org/zip/MM2019DDay',
    ],
    'splits':  {"train":.8,"validation":.1,"test":.1},
}

vacc_itc_config = {
    'cache_dir': data_dir / "raw" / "aifdb" / "vaccitc",
    'templates_sp_ca': [
        'aifdb/source_paraphrase_ca-01.txt',
        'aifdb/source_paraphrase_ca-02.txt',
        'aifdb/source_paraphrase_ca-03.txt',
        'aifdb/source_paraphrase_ca-04.txt',
        'aifdb/source_paraphrase_ca-04.txt',
    ],
    'templates_sp_ra': [
        'aifdb/source_paraphrase_ra-01.txt',
        'aifdb/source_paraphrase_ra-02.txt',
    ],
    'corpora':[
        'http://corpora.aifdb.org/zip/VaccITC1',
        'http://corpora.aifdb.org/zip/VaccITC2',
        'http://corpora.aifdb.org/zip/VaccITC3',
        'http://corpora.aifdb.org/zip/VaccITC4',
        'http://corpora.aifdb.org/zip/VaccITC5',
        'http://corpora.aifdb.org/zip/VaccITC6',
        'http://corpora.aifdb.org/zip/VaccITC7',
        'http://corpora.aifdb.org/zip/VaccITC8',
    ],
    'splits':  {"train":.8,"validation":.1,"test":.1},
}

us2016_config = {
    'cache_dir': data_dir / "raw" / "aifdb" / "us2016",
    'templates_sp_ca': [
        'aifdb/source_paraphrase_ca-01.txt',
        'aifdb/source_paraphrase_ca-02.txt',
        'aifdb/source_paraphrase_ca-03.txt',
        'aifdb/source_paraphrase_ca-04.txt',
        'aifdb/source_paraphrase_ca-04.txt',
    ],
    'templates_sp_ra': [
        'aifdb/source_paraphrase_ra-01.txt',
        'aifdb/source_paraphrase_ra-02.txt',
    ],
    'corpora':[
        'http://corpora.aifdb.org/zip/US2016',
    ],
    'splits':  {"train":.8,"validation":.1,"test":.1},
}
