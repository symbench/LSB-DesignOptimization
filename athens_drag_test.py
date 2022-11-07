import numpy as np
from matplotlib import pyplot as plt

from athens_dragstudy.drag_exploration import fuselage_drag_single_run

FUSE_CYL_LENGTHS = np.arange(100, 1000, 10)  # Should be between 100 - 500

HORIZONTAL_DIAMETER = 140  # Should be between 70 - 300
VERTICAL_DIAMETER = 70     # Should be between 70 - 300
forces_x = []
forces_y = []
forces_z = []


for FUSE_CYL_LENGTH in FUSE_CYL_LENGTHS:
    f = fuselage_drag_single_run({
        "FUSE_CYL_LENGTH": FUSE_CYL_LENGTH,
        "VERT_DIAMETER": VERTICAL_DIAMETER,
        "HORZ_DIAMETER": HORIZONTAL_DIAMETER,
        "BOTTOM_CONNECTOR_ROTATION": 90
    }, direction="xyz")

    forces_x.append(f["drag_force_x"])
    forces_y.append(f["drag_force_y"])
    forces_z.append(f["drag_force_z"])

plt.plot(FUSE_CYL_LENGTHS, forces_x, label="Drag force in x-direction")
plt.plot(FUSE_CYL_LENGTHS, forces_y, label="Drag force in y-direction")
plt.plot(FUSE_CYL_LENGTHS, forces_z, label="Drag force in z-direction")
plt.xlabel("FUSE_CYL_LENGTH")
plt.ylabel("Force at 30 m/s (N)")

plt.legend()
# plt.show()
plt.title("Fuselage drags at various tube lengths.\nVERT_DIAMETER=70, HORZ_DIAMETER=140, ROTATION=90")
plt.tight_layout()
plt.savefig("fuse.png")
