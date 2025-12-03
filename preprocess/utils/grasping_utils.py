import torch
from scipy.spatial.transform import Rotation as Rscipy

def axis_angle_to_rotmat(axis_angle):
    """Converts a torch axis-angle (rotation vector) to a rotation matrix."""
    r = Rscipy.from_rotvec(axis_angle.detach().cpu().numpy())
    return torch.tensor(r.as_matrix(), dtype=torch.float32, device=axis_angle.device)

def rotation_distance(R1, R2):
    """Computes geodesic distance (angle in radians) between two rotation matrices."""
    rel_R = R2 @ R1.T
    r = Rscipy.from_matrix(rel_R.detach().cpu().numpy()).as_rotvec()
    # print(r)
    return torch.tensor(r, dtype=torch.float32, device=R1.device)

def axis_angle_delta(R_curr, R_prev):
    """Returns the relative rotation vector (axis-angle) between two axis-angle vectors."""
    R1 = axis_angle_to_rotmat(R_prev)
    R2 = axis_angle_to_rotmat(R_curr)
    R_rel = R2 @ R1.T
    rotvec = Rscipy.from_matrix(R_rel.detach().cpu().numpy()).as_rotvec()
    return torch.tensor(rotvec, dtype=torch.float32, device=R_curr.device)

def direction_cosine_xy(a, b):
    a_xy = a[..., :2]
    b_xy = b[..., :2]
    a_norm = a_xy / (torch.norm(a_xy) + 1e-6)
    b_norm = b_xy / (torch.norm(b_xy) + 1e-6)
    # print(a_norm, b_norm)
    return torch.dot(a_norm.squeeze(), b_norm.squeeze())

def direction_cosine_3d(a, b):
    """Cosine similarity in full 3D."""
    a_norm = a / (torch.norm(a) + 1e-6)
    b_norm = b / (torch.norm(b) + 1e-6)
    return torch.dot(a_norm.squeeze(), b_norm.squeeze())

def compute_grasping(
    R_hand_r, R_hand_r_prev,       # axis-angle (3,) (unused now)
    R_hand_l, R_hand_l_prev,       # axis-angle (3,) (unused now)
    T_hand_r, T_hand_r_prev,
    T_hand_l, T_hand_l_prev,
    R_obj, R_obj_prev,             # rotation matrices (3x3) (unused now)
    T_obj, T_obj_prev,
    t_thres=1e-3,
    r_thres=1e-2,
    th_thres=5e-3
):
    if R_obj is None or R_obj_prev is None:
        return 0

    # --- Object movement ---
    d_obj_rot = rotation_distance(R_obj, R_obj_prev).abs()
    delta_obj = T_obj[:2] - T_obj_prev[:2]
    # print()
    # print("Object movement:", delta_obj.norm().item(), "Rotation change (rad):", d_obj_rot.norm().item())
    obj_translated = delta_obj.norm() > t_thres
    obj_rotated = d_obj_rot.norm() > r_thres
    obj_moved = obj_translated or obj_rotated

    # obj_moved = d_obj_rot > 0.01

    # --- Hands movement ---
    # print(T_hand_r.shape)
    delta_hand_r = T_hand_r[:2] - T_hand_r_prev[:2]
    delta_hand_l = T_hand_l[:2] - T_hand_l_prev[:2]

    # print(delta_hand_l.norm().item(), delta_hand_r.norm().item())
    left_transl_moved = delta_hand_l.norm() > th_thres
    # left_rot_moved = axis_angle_delta(R_hand_l, R_hand_l_prev).norm() > 1e-2
    left_hand_active = left_transl_moved #or left_rot_moved

    right_transl_moved = delta_hand_r.norm() > th_thres
    # right_rot_moved = axis_angle_delta(R_hand_r, R_hand_r_prev).norm() > 1e-2
    right_hand_active = right_transl_moved #or right_rot_moved

    # --- Translation similarity (XY) ---
    # delta_r_norm = delta_hand_r / (torch.norm(delta_hand_r) + 1e-6)
    # delta_l_norm = delta_hand_l / (torch.norm(delta_hand_l) + 1e-6)
    # delta_obj_norm = delta_obj / (torch.norm(delta_obj) + 1e-6)


    # print("right", delta_obj, delta_hand_r)
    # right_hand_moved = delta_hand_r.norm() > 1e-4
    sim_transl_r = direction_cosine_xy(delta_obj, delta_hand_r)
    # print("left", delta_obj, delta_hand_l)
    sim_transl_l = direction_cosine_xy(delta_obj, delta_hand_l) 

    # --- Rotation similarity (3D) ---
    # delta_rot_obj = Rscipy.from_matrix((R_obj @ R_obj_prev.T).detach().cpu().numpy()).as_rotvec()
    # delta_rot_obj = torch.tensor(delta_rot_obj, dtype=torch.float32, device=R_obj.device)

    # delta_rot_r = axis_angle_delta(R_hand_r, R_hand_r_prev).abs()
    # delta_rot_l = axis_angle_delta(R_hand_l, R_hand_l_prev).abs()

    # sim_rot_r = direction_cosine_3d(delta_rot_obj, delta_rot_r)
    # sim_rot_l = direction_cosine_3d(delta_rot_obj, delta_rot_l)

    # right_movement = delta_hand_r.norm()
    # left_movement = delta_hand_l.norm()

    # right_rot = delta_rot_r.norm()
    # left_rot = delta_rot_l.norm()

    # confidence_r = right_movement * sim_transl_r + right_rot * sim_rot_r if right_hand_active else -1.0
    # confidence_l = left_movement * sim_transl_l + left_rot * sim_rot_l if left_hand_active else -1.0

    # print(confidence_r, confidence_l)

    # print("Right hand movement:", delta_hand_r.norm().item(), "Direction:", delta_r_norm.cpu().numpy())
    # print("Left hand movement:", delta_hand_l.norm().item(), "Direction:", delta_l_norm.cpu().numpy())
    # print("Object movement direction:", delta_obj_norm.cpu().numpy())

    # rotation similarity
    # print("Right hand rotation change (rad):", delta_rot_r)
    # print("Left hand rotation change (rad):", delta_rot_l)
    # print("Object rotation change (rad):", d_obj_rot)

    # --- Confidence scores ---
    # print('similarities:', sim_transl_r.item(), sim_rot_r.item(), sim_transl_l.item(), sim_rot_l.item())
    # confidence_r = 0.7 * sim_transl_r + 0.3 * sim_rot_r if right_hand_active else -1.0
    # confidence_l = 0.7 * sim_transl_l + 0.3 * sim_rot_l if left_hand_active else -1.0

    # check if hand translated or rotated enough and compute weighted confidence only based on this check
    # tr_w, rot_w = 0.5, 0.5 if right_hand_active else 0.0, 
    # if right_transl_moved and right_rot_moved and obj_rotated and obj_translated:
    #     tr_w_r, rot_w_r = 0.5, 0.5
    # elif right_transl_moved and obj_translated:
    #     tr_w_r, rot_w_r = 1.0, 0.0
    # elif right_rot_moved and obj_rotated:
    #     tr_w_r, rot_w_r = 0.0, 1.0
    # else:
    #     tr_w_r, rot_w_r = 0.0, 0.0

    # if left_transl_moved and left_rot_moved and obj_rotated and obj_translated:
    #     tr_w_l, rot_w_l = 0.5, 0.5
    # elif left_transl_moved and obj_translated:
    #     tr_w_l, rot_w_l = 1.0, 0.0
    # elif left_rot_moved and obj_rotated:
    #     tr_w_l, rot_w_l = 0.0, 1.0
    # else:
    #     tr_w_l, rot_w_l = 0.0, 0.0

    # confidence_r = tr_w_r * sim_transl_r + rot_w_r * sim_rot_r if right_hand_active else -1.0
    # confidence_l = tr_w_l * sim_transl_l + rot_w_l * sim_rot_l if left_hand_active else -1.0

    confidence_r = sim_transl_r if right_hand_active else -1.0
    confidence_l = sim_transl_l if left_hand_active else -1.0

    # --- Forced grasping logic ---
    if not obj_moved or (confidence_r < 0.0 and confidence_l < 0.0):
        return 0  # Object is static → no grasping

    # Pick the hand with **higher confidence**
    # print(confidence_r, confidence_l)
    if confidence_r > 0.5 and confidence_l > 0.5:
        return 3
    elif confidence_l >= confidence_r:
        return 2  # Left hand
    else:
        return 1  # Right hand