"""
Visualize a robot at the zero (T-pose) joint configuration with a sphere
attached to its end-effector attachment frame.
"""
from pathlib import Path
import time

from fire import Fire
from scipy.spatial.transform import Rotation

import vamp
from vamp import pybullet_interface as vpb


def main():
    robot_dir = Path(__file__).parent.parent / "resources/motoman"
    vamp_module = vamp.motoman


    # Build attachment: offset along EE z-axis, with specified rotation
    attachment = vamp.Attachment(
        [0, 0, 0],
        [0, 0, 0, 1],
    )
    attachment.add_spheres([
        vamp.Sphere([0, 0, 0], 0.2), 
        vamp.Sphere([0, 0, 0.1], 0.2), 
        vamp.Sphere([0, 0, 0.2], 0.2),
        vamp.Sphere([0, 0, 0.3], 0.2)
    ])

    # Start PyBullet simulator
    sim = vpb.PyBulletSimulator(
        str(robot_dir / f"motoman_spherized.urdf"),
        vamp.ROBOT_JOINTS["motoman"],
        True,
    )
    # Place robot at zero config
    sim.set_joint_positions([0.0] * 16)

    CONFIGS = [
        [0.0] * 16,
        [1.5708, 0, 0.0, 0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.5708]
    ]

    for config in CONFIGS:
        # Overlay VAMP collision spheres (green, semi-transparent)
        for sphere in vamp_module.fk(config):
            sim.add_sphere(sphere.r, [sphere.x, sphere.y, sphere.z], color=[0.0, 0.8, 0.0, 0.35])

        # Compute EE pose at config and set it on the attachment
        position, orientation_xyzw = vamp_module.eefk(config)
        attachment.set_ee_pose(position, orientation_xyzw)
        # Visualize attachment sphere at its posed position
        for sphere in attachment.posed_spheres:
            sim.add_sphere(sphere.r, sphere.position, color=[0.9, 0.2, 0.2, 0.8])
    
    print(f"EE position : {list(round(v, 4) for v in position)}")
    print(f"EE quaternion (xyzw): {list(round(v, 4) for v in orientation_xyzw)}")
    print("Press Ctrl+C to exit.")

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    Fire(main)
