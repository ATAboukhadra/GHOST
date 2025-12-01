import torch
import numpy as np
import matplotlib.colors as mcolors
import trimesh
import cv2
from pytorch3d.renderer import (
    PerspectiveCameras,
    PointLights, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    HardPhongShader,
    Textures
)
from pytorch3d.structures.meshes import Meshes
from pytorch3d.structures import join_meshes_as_scene

def project_3D_points(cam_mat, pts3D):
    '''
    Function for projecting 3d points to 2d
    :param camMat: camera matrix
    :param pts3D: 3D points
    '''

    pts3D[:, :, 2] *= -1
    proj_pts = pts3D @ cam_mat.T
    proj_pts2d = proj_pts[:, :, :2] / proj_pts[:, :, 2:]
    
    return proj_pts2d


class MeshPyTorch3DRenderer:

    def __init__(self, cfg, faces, device, render_res=[256, 256], focal_length=None, mesh_base_color=(1.0, 1.0, 0.9), scene_bg_color=(0, 0, 0)):
        
        self.cfg = cfg
        self.focal_length = focal_length
        self.img_res = render_res
        self.device = device
        self.renderer = self.create_renderer(focal_length, render_res, device)
        self.cam_int = np.array([[float(focal_length), 0, float(render_res[0]) // 2], \
                                    [0, float(focal_length), float(render_res[1]) // 2], \
                                    [0, 0, 1]])
        # add faces that make the hand mesh watertight
        faces_new = np.array([[92, 38, 234],
                              [234, 38, 239],
                              [38, 122, 239],
                              [239, 122, 279],
                              [122, 118, 279],
                              [279, 118, 215],
                              [118, 117, 215],
                              [215, 117, 214],
                              [117, 119, 214],
                              [214, 119, 121],
                              [119, 120, 121],
                              [121, 120, 78],
                              [120, 108, 78],
                              [78, 108, 79]])
        faces = np.concatenate([faces, faces_new], axis=0)
        
        # self.camera_center = [self.img_res // 2, self.img_res // 2]
        self.faces = faces
        self.faces_left = self.faces[:,[0,2,1]]

    def create_renderer(self, focal_length, render_res, device):
        
        render_res = (int(render_res[0]), int(render_res[1]))

        cameras = PerspectiveCameras(
            focal_length=((focal_length, focal_length),),
            principal_point=((render_res[0] / 2, render_res[1] / 2),),
            image_size=((render_res[1], render_res[0]),),
            device=device,
            in_ndc=False,
        )

        # Initialize a switched off light
        lights = PointLights(location=[[0.0, 0.0, -3.0]], device=device)
        raster_settings = RasterizationSettings(
            image_size=(render_res[1], render_res[0]),
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=None
        )
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
            shader=HardPhongShader(device=device, cameras=cameras, lights=lights),
        )
        # Render the mesh and return the output image
        return renderer

    def create_mesh(self, verts, faces, color=(1, 1, 1)):
            
        dummy_verts_uvs = torch.zeros_like(verts[:, :, :2], device=verts.device) 
        r, g, b = color
        dummy_texture_image = torch.tensor([[[[r, g, b], [r, g, b]]]], device=verts.device)  # White dummy texture image        
        tex = Textures(verts_uvs=dummy_verts_uvs, faces_uvs=faces, maps=dummy_texture_image)
        # Create a meshes object with textures
        
        mesh = Meshes(verts=verts, faces=faces, textures=tex)
        return mesh

    def vertices_to_trimesh(self, vertices, camera_translation, mesh_base_color=(1.0, 1.0, 0.9), rot_axis=[1,0,0], rot_angle=0, is_right=1):

        vertex_colors = np.array([(*mesh_base_color, 1.0)] * vertices.shape[0])
        if is_right:
            mesh = trimesh.Trimesh(vertices.copy() + camera_translation, self.faces.copy(), vertex_colors=vertex_colors)
        else:
            mesh = trimesh.Trimesh(vertices.copy() + camera_translation, self.faces_left.copy(), vertex_colors=vertex_colors)
        
        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [0, 0, 1])
        mesh.apply_transform(rot)

        return mesh
    
    def fast_render_rgb_frame_pytorch3d(self, vertices, cam_t, is_right=None):
        if is_right is None:
            is_right = [1 for _ in range(len(vertices))]
        
        meshes = []
        colors = [mcolors.to_rgb('#858AF1'), mcolors.to_rgb('#24788F')]

        for verts, cam_trans, is_right_hand in zip(vertices, cam_t, is_right):
            mesh = self.vertices_to_trimesh(verts, cam_trans, is_right=is_right_hand)
            verts_tensor = torch.tensor(mesh.vertices, dtype=torch.float32, device=self.device)
            faces_tensor = torch.tensor(mesh.faces, dtype=torch.int64, device=self.device)
            meshes.append(self.create_mesh(verts_tensor.unsqueeze(0), faces_tensor.unsqueeze(0), color=colors[int(is_right_hand)]))
        
        meshes = join_meshes_as_scene(meshes)
        rendered_image = self.renderer(meshes)
        rendered_image = rendered_image.cpu().numpy()[0]

        return rendered_image


        
