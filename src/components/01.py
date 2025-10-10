import gc
import sys

# Clear all variables
for name in dir():
    if not name.startswith('_'):
        del globals()[name]

gc.collect()
