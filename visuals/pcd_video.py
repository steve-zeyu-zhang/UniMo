import os
import re
import tempfile
import shutil
import numpy as np
import cv2
import open3d as o3d
import ffmpeg 
from open3d.visualization.rendering import OffscreenRenderer, MaterialRecord

def load_pcd_files(pcd_dir):
    pcd_files = []
    for root, _, files in os.walk(pcd_dir):
        for file in files:
            if file.endswith(".pcd"):
                pcd_files.append(os.path.join(root, file))

    # Improved frame number extraction method
    def extract_frame_number(filepath):
        filename = os.path.basename(filepath)
        # Use regex to match the numeric part in the filename, e.g., _f0000.pcd
        match = re.search(r'_f(\d+)\.pcd$', filename)
        if match:
            return int(match.group(1))
        else:
            # Fallback: match the number at the end of the filename
            matches = re.findall(r'\d+', filename)
            if matches:
                return int(matches[-1])
            else:
                raise ValueError(f"Cannot parse frame number from filename: {filename}")

    pcd_files.sort(key=extract_frame_number)
    return pcd_files

def visualize_point_cloud_offscreen(pcd_files, output_video_path, fps=15, frame_size=(640, 480)):
    """
    apt-get install mesa-utils
    apt-get install xvfb
    pkill Xvfb
    Xvfb :1 -screen 0 640x480x24 &
    export DISPLAY=:1

    Use OffscreenRenderer for offscreen point cloud rendering and save as an H.264 MP4 video using ffmpeg-python.
    
    Parameters:
      pcd_files: List of point cloud files
      output_video_path: Output video file path
      fps: Frames per second
      frame_size: Video frame size (width, height)
    """
    
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    # Create offscreen renderer
    renderer = OffscreenRenderer(frame_size[0], frame_size[1])
    renderer.scene.set_background([0, 0, 0, 1])
    
    # Set material using unlit shader
    material = MaterialRecord()
    material.shader = "defaultUnlit"
    
    # Create a temporary directory to store frame images
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Initialization: read the first point cloud and add it to the scene
        first_pcd = o3d.io.read_point_cloud(pcd_files[0])
        if not first_pcd.has_colors():
            first_pcd.paint_uniform_color([1, 1, 1])
        renderer.scene.add_geometry("pointcloud", first_pcd, material)
        
        # Set camera view: compute center and extent from the point cloud bounding box
        bbox = first_pcd.get_axis_aligned_bounding_box()
        center = bbox.get_center()
        extent = np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound())
        # Set camera position: place it at a distance along the positive Z-axis
        eye = center + np.array([0, 0, extent])
        up = np.array([0, 1, 0])
        renderer.scene.camera.look_at(center, eye, up)
        
        # Iterate through all point cloud files and render each frame
        for idx, pcd_file in enumerate(pcd_files):
            pcd = o3d.io.read_point_cloud(pcd_file)
            if not pcd.has_colors():
                pcd.paint_uniform_color([1, 1, 1])
            
            # For frames after the first, remove the existing geometry and add the new one
            if idx > 0:
                renderer.scene.remove_geometry("pointcloud")
                renderer.scene.add_geometry("pointcloud", pcd, material)
            
            # Render the current frame
            image_o3d = renderer.render_to_image()
            image = np.asarray(image_o3d)
            if image.shape[2] == 4:
                image = image[:, :, :3]
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Save frame to temporary directory
            frame_path = os.path.join(temp_dir, f"frame_{idx:05d}.png")
            cv2.imwrite(frame_path, image)
            # print(f"Frame processed {idx + 1}/{len(pcd_files)}", end='\r')
            
    finally:
        # Clean up the renderer resource
        del renderer

    # Use ffmpeg-python to encode the images into an H.264 video.
    try:
        (
            ffmpeg
            .input(os.path.join(temp_dir, "frame_%05d.png"), framerate=fps)
            .output(output_video_path, vcodec='libx264', pix_fmt='yuv420p')
            .run(overwrite_output=True, quiet=True, capture_stdout=True, capture_stderr=True)
        )
    except Exception as e:
        print("FFmpeg error:", e)
        raise

    # Remove the temporary directory
    shutil.rmtree(temp_dir)
    # print(f"\n Video saved to: {output_video_path}")

if __name__ == "__main__":
    # -------------------------------
    root_dir = r"/root/autodl-tmp/pcmrl-vis/checkpoints/cmu/cmu_debug/visuals"
    skeleton_name = "cmu"
    pcd_type = "predict"
    pcd_root_dir = f"/root/autodl-tmp/pcmrl-vis/checkpoints/cmu/cmu_debug/visuals/pcd/{pcd_type}/cmu"

    pcd_vis_dir = os.path.join(root_dir, "pcd_vis", pcd_type, skeleton_name)
    os.makedirs(pcd_vis_dir, exist_ok=True)
    sample_dirs = sorted([d for d in os.listdir(pcd_root_dir) if d.startswith(f"{pcd_type}_")])

    for sample_dir in sample_dirs:
        sample_path = os.path.join(pcd_root_dir, sample_dir)
        sample_base = sample_dir.rsplit('.bvh', 1)[0]
        output_video_path = os.path.join(pcd_vis_dir, f"{sample_base}.mp4")
        try:
            pcd_files = load_pcd_files(sample_path)
            if not pcd_files:
                print(f"Skipping empty folder: {sample_path}")
                continue
            print(f"Generating video: {skeleton_name}/{sample_dir} -> {output_video_path}")

            visualize_point_cloud_offscreen(
                pcd_files,
                output_video_path,
                fps=20,
                frame_size=(640, 480)
            )
        except Exception as e:
            print(f"Processing failed: {sample_path}, error: {str(e)}")
    # -------------------------------