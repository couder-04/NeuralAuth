"""
vehicle.py
==========

Vehicle context feature generator.

Generated Features
------------------
- vehicle_speed
- engine_running
- location_familiarity
- time_familiarity
- driver_present
- seatbelt_fastened
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict

import numpy as np


# ==========================================================
# Feature Container
# ==========================================================

@dataclass(slots=True)
class VehicleFeatures:

    vehicle_speed: float

    engine_running: int

    location_familiarity: float

    time_familiarity: float

    driver_present: int

    seatbelt_fastened: int

    def to_dict(self):

        return asdict(self)


# ==========================================================
# Generator
# ==========================================================

class VehicleGenerator:

    def __init__(self, rng=None):

        self.rng = rng or np.random.default_rng()

    # ------------------------------------------------------

    def generate(
        self,
        fraudulent: bool = False,
    ) -> VehicleFeatures:

        # --------------------------------------------
        # Driver Present
        # --------------------------------------------

        if fraudulent:

            driver_present = self.rng.choice(
                [0, 1],
                p=[0.25, 0.75],
            )

        else:

            driver_present = self.rng.choice(
                [0, 1],
                p=[0.02, 0.98],
            )

        # --------------------------------------------
        # Engine Running
        # --------------------------------------------

        if fraudulent:

            engine_running = self.rng.choice(
                [0, 1],
                p=[0.20, 0.80],
            )

        else:

            engine_running = self.rng.choice(
                [0, 1],
                p=[0.15, 0.85],
            )

        # --------------------------------------------
        # Vehicle Speed
        # --------------------------------------------

        if engine_running == 0:

            vehicle_speed = 0.0

        else:

            if fraudulent:

                vehicle_speed = np.clip(

                    self.rng.normal(
                        55,
                        25,
                    ),

                    0,
                    140,

                )

            else:

                vehicle_speed = np.clip(

                    self.rng.normal(
                        32,
                        18,
                    ),

                    0,
                    120,

                )

        # --------------------------------------------
        # Location Familiarity
        # --------------------------------------------

        if fraudulent:

            location_familiarity = np.clip(

                self.rng.normal(
                    0.28,
                    0.20,
                ),

                0,
                1,

            )

        else:

            location_familiarity = np.clip(

                self.rng.normal(
                    0.88,
                    0.10,
                ),

                0,
                1,

            )

        # --------------------------------------------
        # Time Familiarity
        # Correlated with location familiarity
        # --------------------------------------------

        time_familiarity = np.clip(

            0.65 * location_familiarity
            + self.rng.normal(
                0.25,
                0.10,
            ),

            0,
            1,

        )

        # --------------------------------------------
        # Seatbelt
        # --------------------------------------------

        if driver_present == 0:

            seatbelt_fastened = 0

        else:

            if fraudulent:

                seatbelt_fastened = self.rng.choice(
                    [0, 1],
                    p=[0.25, 0.75],
                )

            else:

                seatbelt_fastened = self.rng.choice(
                    [0, 1],
                    p=[0.05, 0.95],
                )

        return VehicleFeatures(

            vehicle_speed=round(
                float(vehicle_speed),
                2,
            ),

            engine_running=int(engine_running),

            location_familiarity=round(
                float(location_familiarity),
                4,
            ),

            time_familiarity=round(
                float(time_familiarity),
                4,
            ),

            driver_present=int(driver_present),

            seatbelt_fastened=int(seatbelt_fastened),

        )


# ==========================================================
# Public API
# ==========================================================

_generator = VehicleGenerator()


def generate_vehicle(
    fraudulent: bool = False,
) -> Dict:

    return _generator.generate(
        fraudulent=fraudulent,
    ).to_dict()


# ==========================================================
# Demo
# ==========================================================

if __name__ == "__main__":

    print("\nLegitimate Vehicles\n")

    for _ in range(5):

        print(generate_vehicle())

    print("\nFraudulent Vehicles\n")

    for _ in range(5):

        print(generate_vehicle(fraudulent=True))