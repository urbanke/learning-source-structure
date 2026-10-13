"""The public ``lsa.universal_tables`` under the development name (see the package docstring)."""
from product_model_with_memory import map_settings as _map
_map()
from lsa.universal_tables import *  # noqa: F401,F403
from lsa.universal_tables import UniversalTables  # noqa: F401  (explicit: the class every caller uses)
