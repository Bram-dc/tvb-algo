enabled = True


def randomness_enabled():
    return enabled


def disable_randomness():
    global enabled
    enabled = False
