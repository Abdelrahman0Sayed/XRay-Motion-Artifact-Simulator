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
    compute_psnr,
    compute_snr,
    project_parallel,
    simulate_acquisition,
)


class SimulationWorker(threading.Thread):
    """Compute a simulation in a background thread and publish events."""

    def __init__(self, phantom, params: SimulationParams):
        super().__init__(daemon=True)
        self.phantom = phantom
        self.params = params

    def run(self):
        try:
            p = self.params

            pub.sendMessage(topics.SIM_PROGRESS, message=SimulationProgressMessage(10))
            vol = crop_phantom(self.phantom, p.body_part)

            pub.sendMessage(topics.SIM_PROGRESS, message=SimulationProgressMessage(20))
            static_proj = project_parallel(vol, p.proj_axis)

            pub.sendMessage(topics.SIM_PROGRESS, message=SimulationProgressMessage(30))
            motion_proj = simulate_acquisition(vol, p)

            pub.sendMessage(topics.SIM_PROGRESS, message=SimulationProgressMessage(72))
            static_noisy = add_noise(static_proj, p.noise_type, p.n_photons)
            noisy = add_noise(motion_proj, p.noise_type, p.n_photons)

            pub.sendMessage(topics.SIM_PROGRESS, message=SimulationProgressMessage(83))
            mitigated = apply_mitigation(noisy, p.mitigation)

            pub.sendMessage(topics.SIM_PROGRESS, message=SimulationProgressMessage(95))
            metrics = {
                "snr_motion": compute_snr(static_proj, noisy),
                "snr_mitigated": compute_snr(static_proj, mitigated),
                "psnr_motion": compute_psnr(static_proj, noisy),
                "psnr_mitig": compute_psnr(static_proj, mitigated),
            }

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
