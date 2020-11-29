import os
import json

import tensorflow as tf

from logging.config import dictConfig

from helpers import parse_boolean
from my_tf_model import MYTFMODEL

persistent_storage_path = "/mnt/persistentstorage"
os.makedirs(persistent_storage_path, exist_ok=True)

podName = os.getenv("MY_POD_NAME", "NA")
nodeName = os.getenv("MY_NODE_NAME", "NA")
app_version = os.getenv("MY_VERSION", "NA")
env_string = os.getenv("MY_ENVIRONMENT", "NA")

##### Logger initialisation #####

arro = [persistent_storage_path]
arro.extend(['logs', 'application_pods', '_'.join([podName, nodeName]) + '.log'])
logFilename = os.path.join(*arro)
os.makedirs(os.path.dirname(logFilename), exist_ok=True)

dictConfig({
    'version': 1,
    'disable_existing_loggers' : False,
    'formatters': {
        'default': {
            'format': '{0} {1} in {2}: {3}'.format('[%(asctime)s]', '%(levelname)s', '%(module)s', '%(message)s')
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stdout',
            'level': 'INFO',
            'formatter': 'default'
        },
        'file': {
            'class':'logging.handlers.TimedRotatingFileHandler',
            'filename': logFilename,
            'formatter': 'default',
            'level': 'INFO',
            'when': 'midnight',
            'interval': 1
        }      
    },
    'root': {
        'level': 'INFO',
        'handlers': ['console', 'file']
    }
})

strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

my_model = MYTFMODEL(persistent_storage_path, strategy)

my_model.load_samples()

with strategy.scope():
    my_model.create_model()

# my_model.try_load_previous_training()

my_model.train_model()
my_model.save_model()


