from bird_olsc import run_bird

MAIN_SELECTOR = "fll"
MAIN_SELECTOR_MULTIPLIER = 128.0
MAIN_VERSION = "BIRD-FLL128-Restartable"


def run_main_bird(actor, seed=0, T=5000, inventory=200, ctx=None, trajectories=None, rewards=None):
    return run_bird(
        actor,
        seed=seed,
        selector_name=MAIN_SELECTOR,
        selector_multiplier=MAIN_SELECTOR_MULTIPLIER,
        T=T,
        inventory=inventory,
        ctx=ctx,
        trajectories=trajectories,
        rewards=rewards,
    )
