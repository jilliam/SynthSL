import pickle
import json
import sys
from os import walk
root = '.'
sys.path.insert(0, root)

from blender_body import ex as body_ex

recover_json_string = ' '.join(sys.argv[sys.argv.index('--') + 1:])
json_config = json.loads(recover_json_string)


   
r = body_ex.run(config_updates=json_config)
