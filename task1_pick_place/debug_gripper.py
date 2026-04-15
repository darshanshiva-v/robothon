#!/usr/bin/env python3
"""
debug_gripper.py — Visual debugging of SO-101 pick-place in MuJoCo.
Shows arm, cube, target, and waypoints. Helps understand gripper geometry.
"""

import numpy as np
import mujoco
import mujoco.viewer as viewer
import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_XML = os.path.join(_SCRIPT_DIR, "..", "workshop", "dev", "docker", "workspace", "src", "so101_mujoco", "mujoco", "scene.xml")

def main():
    model = mujoco.MjModel.from_xml_path(_XML)
    data  = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # IDs
    gripper_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
    cube_body    = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "red_box")
    fixed_id     = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "fixed_jaw_box")
    moving_id    = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "moving_jaw_box")
    site_id      = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
    cube_jid     = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "red_box_joint")

    cube_geom    = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "red_box_geom")
    cube_pos     = data.xpos[cube_body]
    cube_top     = cube_pos[2] + model.geom_size[cube_geom][2]
    target_pos   = np.array([0.30, 0.10, 0.0])

    print("Starting interactive viewer...")
    print("Controls:")
    print("  SPACE - step once")
    print("  ENTER - run full episode")
    print("  Q     - quit")
    print()

    v = viewer.launch_passive(model, data)

    def set_joints(q, gripper_val=0.0):
        names = ["shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll"]
        for i, name in enumerate(names):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            data.qpos[model.jnt_qposadr[jid]] = q[i]
        data.qpos[5] = gripper_val
        mujoco.mj_forward(model, data)

    def get_state():
        mujoco.mj_forward(model, data)
        gb  = data.xpos[gripper_body]
        fx  = data.geom_xpos[fixed_id]
        mx  = data.geom_xpos[moving_id]
        cp  = data.xpos[cube_body]
        R   = data.xmat[gripper_body].reshape(3, 3)
        gf  = gb + R @ model.site_pos[site_id]
        return dict(gb=gb, gf=gf, fx=fx, mx=mx, cp=cp)

    def reset_scene():
        mujoco.mj_resetData(model, data)
        qa = model.jnt_qposadr[cube_jid]
        data.qpos[qa:qa+3] = [0.15, 0.0, 0.02]
        data.qpos[qa+3:qa+7] = [1, 0, 0, 0]
        mujoco.mj_forward(model, data)

    def print_info(s=""):
        st = get_state()
        print(f"{s}")
        print(f"  Gripper body:     {st['gb'].round(4)}")
        print(f"  Gripperframe:     {st['gf'].round(4)}")
        print(f"  Fixed jaw center:  {st['fx'].round(4)}, bot: {st['fx'][2]-0.02:.4f}")
        print(f"  Moving jaw center: {st['mx'].round(4)}, bot: {st['mx'][2]-0.02:.4f}")
        print(f"  Cube body:         {st['cp'].round(4)}, top: {cube_top:.4f}")
        print(f"  Target:            {target_pos}")
        print(f"  Cube→Target error: {np.linalg.norm(st['cp'][:2] - target_pos[:2])*1000:.1f}mm")

    # ── Test modes ──────────────────────────────────────────────────────────────

    def test_joint_config(q, gripper_val=0.0, label=""):
        reset_scene()
        set_joints(q, gripper_val)
        print_info(f"[{label}] joints={[round(x,3) for x in q]} + grip={gripper_val}")
        input("  Press ENTER to apply force...")
        # Apply downward force to see if cube lifts
        for _ in range(200):
            mujoco.mj_applyFT(model, data, [0, 0, -2, 0, 0, 0],
                             data.xpos[gripper_body], gripper_body, data)
            mujoco.mj_step(model, data)
        print(f"  After force: cube={get_state()['cp'].round(4)}")

    def run_waypoint_episode():
        reset_scene()
        # New waypoints (IK-verified for cube at [0.15, 0, 0.02])
        wps = [
            [0.0,  -0.40,  0.20,  0.10, 0.0,  0.0],   # 0: home
            [0.0,  -1.60,  1.64,  0.30, 0.0,  0.0],   # 1: above cube
            [0.0,  -1.15,  1.64,  0.30, 0.0,  0.0],   # 2: descend to grasp
            [0.0,  -1.15,  1.64,  0.30, 0.0, -0.17],  # 3: close
            [0.0,  -1.60,  1.64,  0.30, 0.0, -0.17],  # 4: lift
            [0.0,  -0.30,  0.61,  0.30, 0.0, -0.17],  # 5: swing to target
            [0.0,  -0.20,  0.66,  0.30, 0.0, -0.17],  # 6: descend
            [0.0,  -0.20,  0.66,  0.30, 0.0,  0.0],   # 7: release
            [0.0,  -0.30,  0.61,  0.30, 0.0,  0.0],   # 8: lift
            [0.0,  -0.40,  0.20,  0.10, 0.0,  0.0],   # 9: home
        ]
        PHASE_LEN = 35
        for step in range(400):
            phase = min(step // PHASE_LEN, 9)
            t = (step % PHASE_LEN) / PHASE_LEN
            wp_a = np.array(wps[phase])
            wp_b = np.array(wps[(phase + 1) % 10])
            target = wp_a + (wp_b - wp_a) * t
            set_joints(target[:5], target[5])
            for _ in range(5):
                mujoco.mj_step(model, data)
            if step % 35 == 0:
                st = get_state()
                err = np.linalg.norm(st["cp"][:2] - target_pos[:2]) * 1000
                print(f"  WP{phase}: cube={st['cp'].round(4)}, error={err:.1f}mm")
            v.sync()
        print_info("FINAL")
        input("  Press ENTER to continue...")

    def test_grid_approach():
        """Find approach configs by grid search, display them visually."""
        reset_scene()
        targets = [
            ("Cube XY, Z=0.10",  np.array([cube_pos[0], cube_pos[1], 0.10])),
            ("Cube XY, Z=0.06",  np.array([cube_pos[0], cube_pos[1], 0.06])),
            ("Cube XY, Z=0.05",  np.array([cube_pos[0], cube_pos[1], 0.05])),
            ("Target XY, Z=0.10", np.array([0.30, 0.10, 0.10])),
            ("Target XY, Z=0.06", np.array([0.30, 0.10, 0.06])),
        ]
        for name, target in targets:
            print(f"\n=== Searching: {name} ===")
            best = (float("inf"), None, None)
            for sp in np.linspace(-0.5, 0.5, 25):
                for sl in np.linspace(-1.745, -0.2, 25):
                    for ef in np.linspace(0.3, 2.5, 25):
                        q = [sp, sl, ef, 0.30, 0.0]
                        set_joints(q, 0.0)
                        R  = data.xmat[gripper_body].reshape(3, 3)
                        gf = data.xpos[gripper_body] + R @ model.site_pos[site_id]
                        dist = np.linalg.norm(gf - target)
                        if dist < best[0]:
                            best = (dist, list(q), gf.copy())
            print(f"  Best: joints={[round(x,3) for x in best[1]]}, "
                  f"gripperframe={best[2].round(4)}, dist={best[0]:.4f}")
            # Show this config
            set_joints(best[1], -0.17)
            st = get_state()
            print(f"  Gripper body: {st['gb'].round(4)}")
            print(f"  Fixed jaw:    {st['fx'].round(4)}, bot: {st['fx'][2]-0.02:.4f}")
            print(f"  Moving jaw:   {st['mx'].round(4)}, bot: {st['mx'][2]-0.02:.4f}")
            print(f"  Cube top:     {cube_top:.4f}")

    # ── Interactive menu ─────────────────────────────────────────────────────────

    while v.is_running():
        print("\n=== Debug Menu ===")
        print("1: Test original waypoints (collect_demos.py)")
        print("2: Test NEW waypoints (IK-verified)")
        print("3: Grid search approach poses")
        print("4: Test specific joint configs")
        print("5: Run full waypoint episode (visual)")
        print("Q: Quit")
        choice = input("\nChoice: ").strip()

        if choice == "1":
            # Original waypoints
            wps = [
                [0.0, -0.40, 0.20, 0.10, 0.0, 0.0],
                [0.12, -0.60, 0.50, 0.15, 0.0, 0.0],
                [0.12, -1.20, 1.00, 0.25, 0.0, 0.0],
                [0.12, -1.40, 1.20, 0.30, 0.0, -0.17],
                [0.12, -1.00, 0.80, 0.20, 0.0, -0.17],
                [0.38, -0.80, 0.60, 0.15, 0.0, -0.17],
                [0.38, -1.20, 1.00, 0.25, 0.0, -0.17],
                [0.38, -1.20, 1.00, 0.25, 0.0, 0.0],
                [0.38, -0.80, 0.60, 0.15, 0.0, 0.0],
                [0.0, -0.40, 0.20, 0.10, 0.0, 0.0],
            ]
            reset_scene()
            PHASE_LEN = 35
            for step in range(400):
                phase = min(step // PHASE_LEN, 9)
                t = (step % PHASE_LEN) / PHASE_LEN
                wp_a = np.array(wps[phase])
                wp_b = np.array(wps[(phase + 1) % 10])
                target = wp_a + (wp_b - wp_a) * t
                set_joints(target[:5], target[5])
                for _ in range(5):
                    mujoco.mj_step(model, data)
                v.sync()
            st = get_state()
            print(f"Original WP result: cube={st['cp'].round(4)}, "
                  f"error={np.linalg.norm(st['cp'][:2]-target_pos[:2])*1000:.1f}mm")

        elif choice == "2":
            run_waypoint_episode()

        elif choice == "3":
            test_grid_approach()

        elif choice == "4":
            configs = [
                ("sp=0,sl=-1.2,ef=1.2,wf=0.3,grip=0",  [0.0, -1.2, 1.2, 0.3, 0.0], 0.0),
                ("sp=0,sl=-1.2,ef=1.2,wf=0.3,grip=-0.17", [0.0, -1.2, 1.2, 0.3, 0.0], -0.17),
                ("sp=0,sl=-1.5,ef=1.4,wf=0.3,grip=-0.17", [0.0, -1.5, 1.4, 0.3, 0.0], -0.17),
                ("sp=0.1,sl=-1.4,ef=1.3,wf=0.2,grip=-0.17", [0.1, -1.4, 1.3, 0.2, 0.0], -0.17),
                ("sp=0,sl=-1.6,ef=1.6,wf=0.25,grip=-0.17", [0.0, -1.6, 1.6, 0.25, 0.0], -0.17),
                ("sp=-0.1,sl=-1.3,ef=1.2,wf=0.2,grip=-0.17", [-0.1, -1.3, 1.2, 0.2, 0.0], -0.17),
            ]
            for label, q, grip in configs:
                test_joint_config(q, grip, label)

        elif choice.upper() == "Q":
            break

    v.close()
    print("Done.")


if __name__ == "__main__":
    main()