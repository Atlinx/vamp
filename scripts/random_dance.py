from pathlib import Path
import time
import numpy as np

from fire import Fire

import vamp
from vamp import pybullet_interface as vpb
from scipy.spatial.transform import Rotation


def sample_valid(vamp_module, rng):
    while True:
        config = rng.next()
        if vamp_module.validate(config):
            return config


def config_to_list(config):
    if isinstance(config, list):
        return config
    if isinstance(config, np.ndarray):
        return config.tolist()
    return config.to_list()


def make_sphere_overlay(sim, vamp_module, config, color=(0.0, 0.8, 0.0, 0.35)):
    sphere_ids = []
    for sphere in vamp_module.fk(config_to_list(config)):
        sphere_id = sim.add_sphere(
            sphere.r, [sphere.x, sphere.y, sphere.z], color=color
        )
        sphere_ids.append(sphere_id)
    return sphere_ids


def play_once_with_spheres(sim, vamp_module, plan, sphere_ids):
    if not len(plan):
        print(
            """Path has no states!
            """
        )
        return

    for state in plan:
        state_list = config_to_list(state)
        sim.set_joint_positions(state_list)
        for sphere_id, sphere in zip(sphere_ids, vamp_module.fk(state_list)):
            sim.update_object_position(sphere_id, [sphere.x, sphere.y, sphere.z])
        time.sleep(0.016)


def update_attachment_visualization(
    attachment_sphere_id,
    vamp_module,
    sim,
    config,
    attachment,
):
    position, orientation_xyzw = vamp_module.eefk(config)
    attachment.set_ee_pose(position, orientation_xyzw)
    posed_spheres = attachment.posed_spheres
    if posed_spheres:
        sphere = posed_spheres[0]
        sim.update_object_position(
            attachment_sphere_id,
            [sphere.x, sphere.y, sphere.z],
        )


def play_once_with_overlays(
    sim,
    vamp_module,
    plan,
    sphere_ids,
    attachment=None,
    attachment_sphere_id=None,
):
    if not len(plan):
        print(
            """Path has no states!
            """
        )
        return

    for state in plan:
        state_list = config_to_list(state)
        sim.set_joint_positions(state_list)

        for sphere_id, sphere in zip(sphere_ids, vamp_module.fk(state_list)):
            sim.update_object_position(sphere_id, [sphere.x, sphere.y, sphere.z])

        if attachment is not None and attachment_sphere_id is not None:
            update_attachment_visualization(
                attachment_sphere_id,
                vamp_module,
                sim,
                state_list,
                attachment,
            )

        time.sleep(0.016)


def main(
    robot: str = "panda",  # Robot to plan for
    planner: str = "rrtc",  # Planner name to use
    sampler_name: str = "halton",  # Sampler to use.
    skip_rng_iterations: int = 0,  # Skip a number of RNG iterations
    show_spherized_collision_spheres: bool = True,  # Overlay FK collision spheres
    add_ee_attachment: bool = True,  # Add collision sphere attachment at end-effector
    attachment_radius: float = 0.1,  # Radius of end-effector attachment sphere
    attachment_offset: float = 0.05,  # Z offset in end-effector frame for attachment sphere
    **kwargs,
):

    if robot not in vamp.ROBOT_JOINTS:
        raise RuntimeError(f"Robot {robot} does not exist in VAMP!")

    robot_dir = Path(__file__).parent.parent / "resources" / robot

    print(str(robot_dir / f"{robot}_spherized.urdf"))

    (vamp_module, planner_func, plan_settings, simp_settings) = (
        vamp.configure_robot_and_planner_with_kwargs(
            robot,
            planner,
            **kwargs,
        )
    )

    sampler = getattr(vamp_module, sampler_name)()
    sampler.skip(skip_rng_iterations)

    start = sample_valid(vamp_module, sampler).to_list()
    goal = sample_valid(vamp_module, sampler).to_list()
    env = vamp.Environment()

    obstacles = [
        {
            "type": "cuboid",
            "center": (-0.5, -0.5, 2),
            "euler": (0, 45, 45),
            "half_extents": (0.2, 0.2, 0.2),
        },
        {
            "type": "cuboid",
            "center": (-0.5, 0.5, 1),
            "euler": (0, 45, 0),
            "half_extents": (0.2, 0.2, 0.2),
        },
        {
            "type": "sphere",
            "center": (0.5, 0.5, 1.5),
            "radius": 0.25,
        },
    ]

    sim = vpb.PyBulletSimulator(
        str(robot_dir / f"{robot}_spherized.urdf"), vamp.ROBOT_JOINTS[robot], True
    )
    sphere_overlay_ids = None
    if show_spherized_collision_spheres:
        sphere_overlay_ids = make_sphere_overlay(sim, vamp_module, start)

    attachment = None
    attachment_sphere_id = None
    if add_ee_attachment:
        attachment = vamp.Attachment([0, 0, 0], [0, 0, 0, 1])
        attachment.add_spheres([vamp.Sphere([0, 0, 0.2], attachment_radius)])
        env.attach(attachment)

        attachment_sphere_id = sim.add_sphere(
            attachment_radius,
            [0, 0, 0],
            color=[0.9, 0.2, 0.2, 0.8],
        )
        update_attachment_visualization(
            attachment_sphere_id,
            vamp_module,
            sim,
            start,
            attachment,
        )

    for obstacle in obstacles:
        if obstacle["type"] == "cuboid":
            env.add_cuboid(
                vamp.Cuboid(
                    obstacle["center"], obstacle["euler"], obstacle["half_extents"]
                )
            )
            quat = Rotation.from_euler("xyz", obstacle["euler"], degrees=True).as_quat()
            sim.add_cuboid(obstacle["half_extents"], obstacle["center"], quat)
        elif obstacle["type"] == "sphere":
            env.add_sphere(vamp.Sphere(obstacle["center"], obstacle["radius"]))
            sim.add_sphere(obstacle["radius"], obstacle["center"])

    while True:
        result = planner_func(start, goal, env, plan_settings, sampler)
        solved = result.solved
        print(solved)

        if solved:
            simplify = vamp_module.simplify(result.path, env, simp_settings, sampler)
            stats = vamp.results_to_dict(result, simplify)
            print(
                f"""
Planning Time: {stats['planning_time'].microseconds:8d}μs
Simplify Time: {stats['simplification_time'].microseconds:8d}μs
   Total Time: {stats['total_time'].microseconds:8d}μs

Planning Iters: {stats['planning_iterations']}
n Graph States: {stats['planning_graph_size']}

Path Length:
   Initial: {stats['initial_path_cost']:5.3f}
Simplified: {stats['simplified_path_cost']:5.3f}"""
            )

            plan = simplify.path
            plan.interpolate_to_resolution(vamp_module.resolution())

            if show_spherized_collision_spheres and sphere_overlay_ids is not None:
                play_once_with_overlays(
                    sim,
                    vamp_module,
                    plan,
                    sphere_overlay_ids,
                    attachment,
                    attachment_sphere_id,
                )
            else:
                if (
                    add_ee_attachment
                    and attachment is not None
                    and attachment_sphere_id is not None
                ):
                    play_once_with_overlays(
                        sim,
                        vamp_module,
                        plan,
                        [],
                        attachment,
                        attachment_sphere_id,
                    )
                else:
                    sim.play_once(plan)

            start = goal
            goal = sample_valid(vamp_module, sampler).to_list()
        else:
            print("Failed to solve")
            time.sleep(1)
            goal = sample_valid(vamp_module, sampler).to_list()
            continue


if __name__ == "__main__":

    Fire(main)