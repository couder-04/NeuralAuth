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
from typing import Dict, Optional

import numpy as np

from generators.fraud import FraudContext, blend


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
        fraud_context: Optional[FraudContext] = None,
    ) -> VehicleFeatures:

        ctx = FraudContext.resolve(fraudulent, fraud_context)

        # --------------------------------------------
        # Driver Present
        # --------------------------------------------

        impact_driver = ctx.feature_impact("vehicle", "driver_present")
        p_driver_absent = blend(0.02, 0.25, impact_driver)

        driver_present = self.rng.choice(
            [0, 1],
            p=[p_driver_absent, 1.0 - p_driver_absent],
        )

        # --------------------------------------------
        # Engine Running
        # --------------------------------------------

        impact_engine = ctx.feature_impact("vehicle", "engine_running")
        p_engine_off = blend(0.15, 0.20, impact_engine)

        engine_running = self.rng.choice(
            [0, 1],
            p=[p_engine_off, 1.0 - p_engine_off],
        )

        # --------------------------------------------
        # Vehicle Speed
        # --------------------------------------------

        if engine_running == 0:

            vehicle_speed = 0.0

        else:

            impact_speed = ctx.feature_impact("vehicle", "vehicle_speed")

            vehicle_speed = np.clip(

                self.rng.normal(
                    blend(32, 55, impact_speed),
                    blend(18, 25, impact_speed),
                ),

                0,
                blend(120, 140, impact_speed),

            )

        # --------------------------------------------
        # Location Familiarity
        # --------------------------------------------

        impact_loc = ctx.feature_impact("vehicle", "location_familiarity")

        location_familiarity = np.clip(

            self.rng.normal(
                blend(0.88, 0.28, impact_loc),
                blend(0.10, 0.20, impact_loc),
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