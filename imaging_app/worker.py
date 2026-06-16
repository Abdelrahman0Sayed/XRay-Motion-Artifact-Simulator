"""Threaded simulation worker that communicates through pubsub."""

import threading
import traceback

from pubsub import pub

from .messaging import topics
from .messaging.message_types import (
    SimulationDoneMessage,
    SimulationErrorMessage,
    SimulationParams,
    SimulationProgressMessage,
)
from .phantom import crop_phantom
from .physics import (
    add_noise,
    apply_mitigation,
    compute_nmse,  
    compute_ssim,  
    project_parallel,
    project_cone_beam,
    simulate_parallel_acquisition,
    simulate_cone_acquisition,
)


class SimulationWorker(threading.Thread):
    """Compute a simulation in a background thread and publish events."""

    def __init__(self, phantom, params: SimulationParams, projection_type="parallel"):
        super().__init__(daemon=True)
        self.phantom = phantom
        self.params = params
        self.projection_type = projection_type

    def run(self):
        try:
            p = self.params

            pub.sendMessage(topics.SIM_PROGRESS, message=SimulationProgressMessage(10))
            vol = crop_phantom(self.phantom, p.body_part)

            pub.sendMessage(topics.SIM_PROGRESS, message=SimulationProgressMessage(20))
            projection_type = getattr(p, "projection_type", self.projection_type)
            if projection_type == "parallel":
                static_proj = project_parallel(vol, p.proj_axis)
                motion_proj = simulate_parallel_acquisition(vol, p)
                
            elif projection_type == "cone":
                pub.sendMessage(topics.SIM_PROGRESS, message=SimulationProgressMessage(30))
                static_proj = project_cone_beam(vol, p)
                motion_proj = simulate_cone_acquisition(vol, p)
            else:
                static_proj = project_parallel(vol, p.proj_axis)
                motion_proj = simulate_parallel_acquisition(vol, p)


            pub.sendMessage(topics.SIM_PROGRESS, message=SimulationProgressMessage(72))
            static_noisy = add_noise(static_proj, p.noise_type, p.n_photons)
            noisy = add_noise(motion_proj, p.noise_type, p.n_photons)

            pub.sendMessage(topics.SIM_PROGRESS, message=SimulationProgressMessage(83))
            mitigated = apply_mitigation(noisy, p)

            pub.sendMessage(topics.SIM_PROGRESS, message=SimulationProgressMessage(95))
            metrics = {
                "nmse_motion": compute_nmse(static_proj, noisy),
                "nmse_mitigated": compute_nmse(static_proj, mitigated),
                "ssim_motion": compute_ssim(static_proj, noisy),
                "ssim_mitig": compute_ssim(static_proj, mitigated),
            }

            pub.sendMessage(topics.SIM_PROGRESS, message=SimulationProgressMessage(100))

            pub.sendMessage(topics.SIM_PROGRESS, message=SimulationProgressMessage(100))
            pub.sendMessage(
                topics.SIM_DONE,
                message=SimulationDoneMessage(
                    static=static_noisy,
                    motion=noisy,
                    mitigated=mitigated,
                    params=p,
                    metrics=metrics,
                ),
            )
        except Exception:
            pub.sendMessage(
                topics.SIM_ERROR,
                message=SimulationErrorMessage(traceback_text=traceback.format_exc()),
            )
