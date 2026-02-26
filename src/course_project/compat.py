from __future__ import annotations

import sys
import types

from graph_utils import Box

def install_legacy_pickle_aliases() -> None:
    net_mod = sys.modules.setdefault("network", types.ModuleType("network"))
    setattr(net_mod, "Box", Box)

    sim_mod = sys.modules.setdefault("simulator", types.ModuleType("simulator"))

    sim_net = sys.modules.setdefault("simulator.network", types.ModuleType("simulator.network"))
    setattr(sim_net, "Box", Box)
    setattr(sim_mod, "network", sim_net)

    sim_helpers = sys.modules.setdefault("simulator.helpers", types.ModuleType("simulator.helpers"))
    sim_helpers_net = sys.modules.setdefault(
        "simulator.helpers.network", types.ModuleType("simulator.helpers.network")
    )
    setattr(sim_helpers_net, "Box", Box)
    setattr(sim_helpers, "network", sim_helpers_net)
    setattr(sim_mod, "helpers", sim_helpers)
