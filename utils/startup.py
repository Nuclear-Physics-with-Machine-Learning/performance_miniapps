import jax

def should_do_io(_mpi_available, _rank):
    if not _mpi_available or _rank == 0:
        return True
    return False


def set_compute_parameters(local_rank):
    import socket
    try:
        devices = jax.local_devices()
    except:
        devices = []
    if len(devices) == 1:
        # Something external has set which devices are visible
        target_device = devices[0]
    elif len(devices) > 1:
        target_device = devices[local_rank]
    else:
        target_device = jax.devices("cpu")[0]
    # Not a pure function by any means ...
    from jax.config import config; config.update("jax_enable_x64", True)
    # config.update('jax_disable_jit', True)

    return target_device
