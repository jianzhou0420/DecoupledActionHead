from robomimic.config import config_factory
import collections
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.envs.env_base as EB

# You will also need mujoco and trimesh
import mujoco
import trimesh
import numpy as np
import mimicgen
# PART 1: The Mesh Extraction Function (from our previous conversation)
# This function works on raw MuJoCo model and data objects.


shape_meta = {
    "obs": {
        "agentview_image": {
            "shape": [3, 84, 84],
            "type": "rgb"
        },
        "robot0_eye_in_hand_image": {
            "shape": [3, 84, 84],
            "type": "rgb"
        },
        "robot0_eef_pos": {
            "shape": [3]
        },
        "robot0_eef_quat": {
            "shape": [4]
        },
        "robot0_gripper_qpos": {
            "shape": [2]
        },
    },
    "action": {
        "shape": [10]
    },
}


def create_env(env_meta, shape_meta, enable_render=True):
    modality_mapping = collections.defaultdict(list)
    for key, attr in shape_meta['obs'].items():
        modality_mapping[attr.get('type', 'low_dim')].append(key)
    ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=enable_render,
        use_image_obs=enable_render,
    )
    return env


def get_mujoco_scene_mesh(model: mujoco.MjModel, data: mujoco.MjData) -> trimesh.Trimesh:
    """
    Retrieves and combines all meshes from a single MuJoCo environment state.
    """
    all_scene_meshes = []
    for geom_id in range(model.ngeom):
        if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_MESH:
            continue

        mesh_id = model.geom_dataid[geom_id]
        vert_start = model.mesh_vertadr[mesh_id]
        vert_count = model.mesh_vertnum[mesh_id]
        face_start = model.mesh_faceadr[mesh_id]
        face_count = model.mesh_facenum[mesh_id]

        local_verts = model.mesh_vert[vert_start: vert_start + vert_count]
        faces = model.mesh_face[face_start: face_start + face_count].reshape(-1, 3)

        pos_vector = data.geom_xpos[geom_id]
        rot_matrix = data.geom_xmat[geom_id].reshape(3, 3)

        transformed_verts = np.einsum('ij,kj->ki', rot_matrix, local_verts) + pos_vector

        mesh_object = trimesh.Trimesh(vertices=transformed_verts, faces=faces)
        all_scene_meshes.append(mesh_object)

    if not all_scene_meshes:
        return None
    return trimesh.util.concatenate(all_scene_meshes)


# PART 2: Using Robomimic Utils to Get to the Simulator
def main():

    # --- Step 1: Load Environment Metadata ---
    # Replace this with the path to your robomimic dataset
    dataset_path = "/media/jian/ssd4t/DP/first/data/robomimic/datasets/stack_d1/stack_d1_abs_traj_eePose.hdf5"

    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)

    # --- Step 2: Create the Environment using the provided utils ---
    # We set render=False because we don't need to see the window.
    env = create_env(
        env_meta=env_meta,
        shape_meta=shape_meta,
        enable_render=False
    )

    # --- Step 3: Access the Underlying Simulator ---
    # Check if this is a robosuite environment, which uses MuJoCo
    if not EnvUtils.is_robosuite_env(env=env):
        print("This script is designed for Robosuite environments (MuJoCo backend).")
        return

    # IMPORTANT: Reset the environment to make sure the simulation is initialized
    print("Resetting environment...")
    env.reset()

    # The `env` object is a robomimic wrapper.
    # `env.env` gives us the underlying robosuite environment.
    robosuite_env = env.env

    # The robosuite environment contains the MuJoCo simulation model and data
    mujoco_model = robosuite_env.sim.model
    mujoco_data = robosuite_env.sim.data

    print(f"Successfully accessed MuJoCo model (ngeom={mujoco_model.ngeom}) and data.")

    # --- Step 4: Extract the Mesh Data ---
    print("Extracting scene mesh...")
    scene_mesh = get_mujoco_scene_mesh(mujoco_model, mujoco_data)

    if scene_mesh:
        # Now you have the mesh! You can save it, analyze it, etc.
        output_path = "scene_mesh_export.obj"
        scene_mesh.export(output_path)
        print(f"Successfully extracted and saved the scene mesh to '{output_path}'")
    else:
        print("Could not extract any meshes from the scene.")

    # clean up
    env.env.close()


if __name__ == "__main__":
    main()
