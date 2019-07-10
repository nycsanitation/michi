import os
from pathlib import Path

def init():
    """
    Set the global gomi variables based on environment variables.
    """
    # The home directory for gomi files (data, cache, etc.)
    if 'MICHI_HOME' in os.environ:
        home = Path(os.environ['MICHI_HOME'])
    else:
        home = Path.home() / '.michi'

    if not home.exists():
        home.mkdir(parents=True)

    return home

MICHI_HOME = init()

LION_URL = 'https://www1.nyc.gov/assets/planning/download/zip/data-maps/open-data/nyclion_%s.zip'
