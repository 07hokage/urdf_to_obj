import trimesh
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
import rospkg
import os

# Function to convert URDF origin to transformation matrix
def origin_to_transform(xyz, rpy):
    translation = trimesh.transformations.translation_matrix(xyz)
    rotation = trimesh.transformations.euler_matrix(rpy[0], rpy[1], rpy[2], axes='sxyz')
    return translation @ rotation

# Function to resolve package:// paths to absolute paths
def resolve_package_path(filename, mesh_dir):
    if filename.startswith('package://'):
        package_name = filename.split('/')[2]
        relative_path = '/'.join(filename.split('/')[3:])
        try:
            rp = rospkg.RosPack()
            package_path = rp.get_path(package_name)
            return str(Path(package_path) / relative_path)
        except rospkg.ResourceNotFound:
            print(f"Warning: Package {package_name} not found, using mesh_dir")
            return str(Path(mesh_dir) / relative_path)
    return str(Path(mesh_dir) / filename)

# Function to extract meshes from a trimesh.Scene or Trimesh
def get_meshes_from_load(obj):
    if isinstance(obj, trimesh.Trimesh):
        return [obj]
    elif isinstance(obj, trimesh.Scene):
        meshes = []
        for node in obj.geometry.values():
            if isinstance(node, trimesh.Trimesh):
                meshes.append(node)
        return meshes
    else:
        return []

# Function to parse URDF and generate OBJ for specified links relative to a reference link
def generate_obj_from_links(
    urdf_path,
    link_names,
    reference_link,
    output_obj='combined.obj',
    mesh_dir=None,
    gripper_state='normal'
):
    # Validate gripper_state
    valid_states = ['open', 'closed', 'normal']
    if gripper_state not in valid_states:
        print(f"Error: Invalid gripper_state '{gripper_state}'. Must be one of {valid_states}")
        return

    # If mesh_dir is not provided, try to infer from ROS package
    if mesh_dir is None:
        try:
            rp = rospkg.RosPack()
            mesh_dir = str(Path(rp.get_path('fetch_ros_IRVL')) / 'meshes')
        except rospkg.ResourceNotFound:
            print("Error: mesh_dir not provided and fetch_ros_IRVL package not found")
            return

    # Ensure mesh_dir exists
    if not os.path.exists(mesh_dir):
        print(f"Error: mesh_dir {mesh_dir} does not exist")
        return

    # Parse URDF
    try:
        tree = ET.parse(urdf_path)
        root = tree.getroot()
    except Exception as e:
        print(f"Error parsing URDF {urdf_path}: {str(e)}")
        return

    # Collect joint and link information
    joints = {}
    visual_meshes = {}
    visual_origins = {}

    for joint in root.findall('joint'):
        name = joint.get('name')
        parent = joint.find('parent').get('link')
        child = joint.find('child').get('link')
        origin = joint.find('origin')
        xyz = [float(x) for x in origin.get('xyz', '0 0 0').split()] if origin is not None else [0, 0, 0]
        rpy = [float(x) for x in origin.get('rpy', '0 0 0').split()] if origin is not None else [0, 0, 0]

        # Adjust finger joint transformations based on gripper_state
        if child == 'l_gripper_finger_link' and gripper_state != 'normal':
            if gripper_state == 'open':
                xyz[1] -= 0.05  # Move -0.05 along Y-axis (negative Y direction)
            if gripper_state == 'closed':
                xyz[1] -= 0.015  # Move -0.005 along Y-axis
        if child == 'r_gripper_finger_link' and gripper_state != 'normal':
            if gripper_state == 'open':
                xyz[1] += 0.05  # Move +0.05 along Y-axis (positive Y direction)
            if gripper_state == 'closed':
                xyz[1] += 0.015  # Move +0.005 along Y-axis

        joints[child] = {
            'parent': parent,
            'transform': origin_to_transform(xyz, rpy)
        }

    for link in root.findall('link'):
        name = link.get('name')
        if name in link_names:
            visual = link.find('visual')
            if visual is not None:
                geometry = visual.find('geometry/mesh')
                if geometry is not None:
                    filename = geometry.get('filename')
                    if filename:
                        visual_meshes[name] = resolve_package_path(filename, mesh_dir)
                origin = visual.find('origin')
                xyz = [float(x) for x in origin.get('xyz', '0 0 0').split()] if origin is not None else [0, 0, 0]
                rpy = [float(x) for x in origin.get('rpy', '0 0 0').split()] if origin is not None else [0, 0, 0]
                visual_origins[name] = origin_to_transform(xyz, rpy)
            else:
                print(f"Warning: No visual geometry for link {name}")

    # Build transformation hierarchy relative to reference_link
    def get_full_transform(link_name):
        if link_name == reference_link:
            return np.eye(4)  # Identity transform for reference link
        transform = np.eye(4)
        current_link = link_name
        while current_link in joints and current_link != reference_link:
            transform = joints[current_link]['transform'] @ transform
            current_link = joints[current_link]['parent']
        if current_link != reference_link:
            print(f"Warning: Link {link_name} is not a descendant of {reference_link}")
            return np.eye(4)  # Return identity if not connected to reference
        return transform

    # Load and transform meshes
    meshes = []
    for link_name in link_names:
        if link_name not in visual_meshes:
            print(f"Warning: No mesh found for link {link_name}")
            continue
        try:
            # Load mesh
            loaded_obj = trimesh.load(visual_meshes[link_name], force='mesh')
            mesh_list = get_meshes_from_load(loaded_obj)
            
            if not mesh_list:
                print(f"Warning: No valid meshes found for {link_name} ({visual_meshes[link_name]})")
                continue
            
            # Apply hierarchical joint transform
            joint_transform = get_full_transform(link_name)
            
            # Apply visual origin transform
            visual_transform = visual_origins.get(link_name, np.eye(4))
            
            # Combine transforms
            transform = joint_transform @ visual_transform
            
            # Apply transformation to each mesh
            for mesh in mesh_list:
                if mesh.vertices.any():
                    mesh.apply_transform(transform)
                    meshes.append(mesh)
                    print(f"Loaded and transformed {link_name} ({visual_meshes[link_name]})")
                else:
                    print(f"Warning: Mesh {link_name} ({visual_meshes[link_name]}) is empty")
        except Exception as e:
            print(f"Error loading {link_name} ({visual_meshes[link_name]}): {str(e)}")

    # Combine meshes
    if meshes:
        combined = trimesh.util.concatenate(meshes)
        # Export to OBJ (no textures)
        combined.export(output_obj, include_texture=False)
        print(f"Generated {output_obj}")
    else:
        print("Error: No valid meshes loaded")

# Example usage
if __name__ == "__main__":
    urdf_path = '/home/ash/irvl/xpeng/trajectory-tracking-optimization/mm_ws/src/fetch_ros_IRVL/fetch_description/robots/fetch.urdf'
    link_names = [
        'gripper_link',
        'l_gripper_finger_link',
        'r_gripper_finger_link'
    ]
    reference_link = 'wrist_roll_link'
    mesh_dir = '/home/ash/irvl/xpeng/trajectory-tracking-optimization/mm_ws/src/fetch_ros_IRVL/fetch_description'
    generate_obj_from_links(
        urdf_path,
        link_names,
        reference_link,
        output_obj='gripper_combined_wrt_wrist_roll.obj',
        mesh_dir=mesh_dir,
        gripper_state='open'  # Options: 'open', 'closed', 'normal'
    )
