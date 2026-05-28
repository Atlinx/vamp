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


def main(
    robot: str = "motoman",
    attachment_radius: float = 0.1,
    attachment_offset: float = 0.1,  # Z offset in EE frame
    attachment_rotation: Rotation = Rotation.from_euler("xyz", [0, 0, 0], degrees=True),
):
    if robot not in vamp.ROBOT_JOINTS:
        raise RuntimeError(f"Robot {robot} does not exist in VAMP!")

    robot_dir = Path(__file__).parent.parent / "resources" / robot
    vamp_module = getattr(vamp, robot)

    zero_config = [0.0] * vamp_module.dimension()

    # Build attachment: offset along EE z-axis, with specified rotation
    attachment = vamp.Attachment(
        [0, 0, 0],
        attachment_rotation.as_quat(),
    )
    attachment.add_spheres([vamp.Sphere([0, 0, attachment_offset], attachment_radius)])

    # Compute EE pose at zero config and set it on the attachment
    position, orientation_xyzw = vamp_module.eefk(zero_config)
    attachment.set_ee_pose(position, orientation_xyzw)

    # Start PyBullet simulator
    sim = vpb.PyBulletSimulator(
        str(robot_dir / f"{robot}_spherized.urdf"),
        vamp.ROBOT_JOINTS[robot],
        True,
    )

    # Place robot at zero config
    sim.set_joint_positions(zero_config)

    # Overlay VAMP collision spheres (green, semi-transparent)
    for sphere in vamp_module.fk(zero_config):
        sim.add_sphere(sphere.r, [sphere.x, sphere.y, sphere.z], color=[0.0, 0.8, 0.0, 0.35])

    # Visualize attachment sphere at its posed position
    posed = attachment.posed_spheres
    if posed:
        s = posed[0]
        sim.add_sphere(attachment_radius, [s.x, s.y, s.z], color=[0.9, 0.2, 0.2, 0.8])
        print(f"Attachment sphere center: ({s.x:.3f}, {s.y:.3f}, {s.z:.3f})")
    else:
        print("Warning: no posed spheres on attachment.")

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
