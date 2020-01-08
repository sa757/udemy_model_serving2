import os

from img_modeling_pipeline.config import config


with open(os.path.join(config.PACKAGE_ROOT, 'VERSION')) as version_file:
    __version__ = version_file.read().strip()
