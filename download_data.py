# Copyright (c) 2026 Ztry8 (AslanD)
# Licensed under the Apache License, Version 2.0 (the "License");
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

import lightkurve as lk
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=lk.LightkurveWarning)

systems = [
    ("Kepler-8", 1),
    ("Kepler-10", 2),
    ("Kepler-20", 3),
    ("Kepler-37", 3),
    ("Kepler-62", 5),
    ("Kepler-69", 2),
    ("Kepler-11", 6),
    ("Kepler-18", 3),
    ("Kepler-36", 2),
    ("Kepler-90", 8),
    ("Kepler-80", 6),
    ("Kepler-33", 5),
    ("Kepler-160", 3),
    ("Kepler-30", 3),
    ("Kepler-47", 3),
]

max_points = 500
output_prefix = "learn"

def make_learn_file(filename, target, planet_count, max_points=500):
    print(f"Downloading {target} ...")
    search = lk.search_lightcurve(target, mission="Kepler")
    lc = search.download().remove_nans()

    time = lc.time.value
    flux = lc.flux.value
    flux = flux / np.median(flux)

    with open(filename, "w") as f:
        for t, fl in zip(time[:max_points], flux[:max_points]):
            f.write(f"{fl:.6f} {t:.6f}\n")
        f.write(f"result {planet_count}\n")
    print(f"{filename} created")

for i, (target, planet_count) in enumerate(systems, start=1):
    filename = f"{output_prefix}{i}.txt"
    make_learn_file(filename, target, planet_count, max_points)

print("All learnX.txt files created successfully!")