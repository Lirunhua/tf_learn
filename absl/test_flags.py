from absl import app
from absl import flags
from absl import logging
import sys
# In machine learning tasks, there are too many parameters needed to be tuned
# so, how to dynamically change the parameters is a big problem

# flags is a instance
flags.DEFINE_string('name', 'lizedong', 'name of a people')
flags.DEFINE_integer('interests', 0, 'number of interests')

# what flags.FLAGS did here ?
# FLAGS = FlagValues() 
# it instantiation a instance of FlagValues.
FLAGS = flags.FLAGS

print(FLAGS)
def main(_):
    version = sys.version_info
    logging.info("Running under python ")
    print(FLAGS.name, FLAGS.interests)

if __name__ == '__main__':
    app.run(main)