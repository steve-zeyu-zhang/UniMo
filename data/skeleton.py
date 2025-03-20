import torch
import numpy as np
from pytorch3d.transforms import matrix_to_euler_angles




class Skeleton:
    def __init__(self, joint_names, joint_hierarchy, joint_offsets, joint_tails, joint_bgroups, end_joints, pairs,rotation_order, skeleton_name,
                 std=-1, n_point=-1, scale=1.0, device="cpu"):
        self.n_joints = len(joint_names)
        self.device = device
        self.scale = scale
        self.name = skeleton_name

        self.std = std
        self.n_point = n_point

        # 1d array, -1 for root
        self.joint_names = joint_names
        self.joint_hierarchy = joint_hierarchy
        # [left hand, right hand, left toe, right toe]
        self.end_joints = torch.tensor([self.joint_names.index(name) for name in end_joints],
                                       dtype=torch.int64, device=device)

        self.rotation_order = rotation_order  

        self.children = [[] for _ in range(self.n_joints)]
        for i, parent in enumerate(self.joint_hierarchy):
            p = parent if isinstance(parent, int) else int(parent)
            if p != -1:
                self.children[p].append(i)

        # [j, 3] tensor
        self.joint_offsets = joint_offsets.clone().to(device) * scale
        self.joint_tails = joint_tails.clone().to(device) * scale

        # [j, n_bgroups] one-hot tensor
        self.joint_bgroups = joint_bgroups.clone().to(device)

        # clip skeleton at end joints & normalise end joint lengths
        joint_lengths = torch.linalg.vector_norm(self.joint_tails, dim=-1)

        for joint in self.end_joints:
            if joint_lengths[joint] < 1e-6:
                self.joint_tails[joint] = self.joint_tails[self.joint_hierarchy[joint]]
            self.joint_tails[joint] /= torch.linalg.vector_norm(self.joint_tails[joint])
            self.joint_tails[joint] *= 0.1

        post_end_joints = self.end_joints.tolist()
        for i, p_idx in enumerate(self.joint_hierarchy):
            if p_idx in post_end_joints:
                self.joint_tails[i] = 0
                self.joint_offsets[i] = 0
                post_end_joints.append(i)

        # joint size for point cloud distribution
        self.joint_lengths = torch.linalg.vector_norm(self.joint_tails, dim=-1)
        self.joint_dist = torch.distributions.Categorical(probs=self.joint_lengths.cpu())

        # disable relative roll when >2 child joints
        n_children = np.zeros(len(self.joint_hierarchy))
        for i in self.joint_hierarchy[1:]:
            n_children[i] += 1
        self.roll_mask = torch.tensor([n_children[j] < 2 for j in range(len(self.joint_hierarchy))],
                                      dtype=torch.bool, device=device)
        self.pairs = torch.tensor(pairs, dtype=torch.int64, device=device)

        self.id_mat = torch.tensor([[1, 0, 0, 0]], dtype=torch.float32, device=device, requires_grad=False)

    def fk(self, root_p, q, local_q=True, normalise=True):
        # root_p: [b, l, 3]
        # q: [b, l, j, 4] or [b, l, j * 4]
        # output: [b, l, j, *]

        q = torch.reshape(q, (q.shape[0], q.shape[1], -1, 4))
        if normalise:
            q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True)

        if local_q:
            heads = []
            glob_q = []
            for i in range(len(self.joint_hierarchy)):
                parent_index = self.joint_hierarchy[i]
                if parent_index == -1:
                    head = root_p
                    _glob_q = q[:, :, i].contiguous()
                else:
                    head = self._qrot(self.joint_offsets[i], glob_q[parent_index])
                    head = head + heads[parent_index].clone().detach()
                    # _glob_q = self._qmul(glob_q[parent_index].clone().detach(), q[:, :, i].contiguous())
                    _glob_q = self._qmul(glob_q[parent_index], q[:, :, i].contiguous())

                heads.append(head)
                glob_q.append(_glob_q)

            heads = torch.stack(heads, dim=2)
            glob_q = torch.stack(glob_q, dim=2)

        else:
            heads = []
            parent_q = q.transpose(0, 2)[self.joint_hierarchy[1:]].transpose(0, 2)
            offsets = self._qrot(self.joint_offsets[1:], parent_q)
            for i in range(len(self.joint_hierarchy)):
                parent_index = self.joint_hierarchy[i]
                if parent_index == -1:
                    head = root_p
                else:
                    # head = offsets[:, :, i - 1] + heads[parent_index].clone().detach()
                    head = offsets[:, :, i - 1] + heads[parent_index]
                heads.append(head)

            heads = torch.stack(heads, dim=2)
            glob_q = q

        return heads, glob_q
    
    def save_as_pcd(self, points, filename):
        header = f"""# .PCD v0.7 - Point Cloud Data file format
                VERSION 0.7
                FIELDS x y z
                SIZE 4 4 4
                TYPE F F F
                COUNT 1 1 1
                WIDTH {points.shape[0]}
                HEIGHT 1
                VIEWPOINT 0 0 0 1 0 0 0
                POINTS {points.shape[0]}
                DATA ascii
                """
        with open(filename, 'w') as f:
            f.write(header)
            # 确保点云数据格式正确
            np.savetxt(f, points, fmt='%.6f', delimiter=' ')

    def save_bvh(self, filename, root_positions, global_rotations, frame_time=1 / 30.0):
        """
        Args:
            filename: 
            root_positions: [n_frames, 3] 
            global_rotations: [n_frames, n_joints, 4] 
            frame_time: 
        """
        import numpy as np
        from pytorch3d.transforms import matrix_to_euler_angles

        root_positions = root_positions.detach().cpu().numpy()
        global_rotations = global_rotations.detach().cpu().numpy()

        local_rotations = []
        for frame_idx in range(global_rotations.shape[0]):
            frame_rot = {}
            for joint_idx in range(self.n_joints):
                parent_idx = self.joint_hierarchy[joint_idx]

                q_global = torch.tensor(global_rotations[frame_idx, joint_idx], dtype=torch.float32)

                if parent_idx == -1:
                    local_rot = q_global
                else:
                    q_parent_inv = torch.tensor([
                        global_rotations[frame_idx, parent_idx][0],
                        -global_rotations[frame_idx, parent_idx][1],
                        -global_rotations[frame_idx, parent_idx][2],
                        -global_rotations[frame_idx, parent_idx][3]
                    ])

                    # q_parent^{-1} * q_global
                    local_rot = self._qmul(q_parent_inv, q_global)

                frame_rot[joint_idx] = local_rot.numpy()
            local_rotations.append(frame_rot)

        euler_angles = []
        for joint_idx in range(self.n_joints):
            joint_euler = []
            for frame_idx in range(global_rotations.shape[0]):
                q = torch.tensor(local_rotations[frame_idx][joint_idx])
                q_normalized = q / torch.norm(q)

                matrix = torch.zeros(3, 3)
                w, x, y, z = q_normalized
                matrix[0, 0] = 1 - 2 * y ** 2 - 2 * z ** 2
                matrix[0, 1] = 2 * x * y - 2 * z * w
                matrix[0, 2] = 2 * x * z + 2 * y * w
                matrix[1, 0] = 2 * x * y + 2 * z * w
                matrix[1, 1] = 1 - 2 * x ** 2 - 2 * z ** 2
                matrix[1, 2] = 2 * y * z - 2 * x * w
                matrix[2, 0] = 2 * x * z - 2 * y * w
                matrix[2, 1] = 2 * y * z + 2 * x * w
                matrix[2, 2] = 1 - 2 * x ** 2 - 2 * y ** 2


                euler = torch.rad2deg(matrix_to_euler_angles(matrix, self.rotation_order))
                joint_euler.append(euler.numpy())
            euler_angles.append(np.array(joint_euler))

        with open(filename, 'w') as f:
            f.write("HIERARCHY\n")

            def write_joint(joint_idx, parent_idx, indent_level=0):
                indent = "  " * indent_level
                joint_name = self.joint_names[joint_idx]

                if parent_idx == -1:
                    f.write(f"{indent}ROOT {joint_name}\n")
                else:
                    f.write(f"{indent}JOINT {joint_name}\n")

                f.write(f"{indent}{{\n")
                indent += "  "

                offset = (self.joint_offsets[joint_idx].cpu().numpy() / self.scale).tolist()
                f.write(f"{indent}OFFSET {offset[0]:.6f} {offset[1]:.6f} {offset[2]:.6f}\n")

                if parent_idx == -1:
                    f.write(f"{indent}CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation\n")
                else:
                    f.write(f"{indent}CHANNELS 3 Zrotation Yrotation Xrotation\n")

                if len(self.children[joint_idx]) > 0:
                    for child_idx in self.children[joint_idx]:
                        write_joint(child_idx, joint_idx, indent_level + 1)
                else:
                    f.write(f"{indent}End Site\n")
                    f.write(f"{indent}{{\n")
                    indent += "  "
                    end_offset = (self.joint_tails[joint_idx].cpu().numpy() / self.scale).tolist()
                    f.write(f"{indent}OFFSET {end_offset[0]:.6f} {end_offset[1]:.6f} {end_offset[2]:.6f}\n")
                    indent = indent[:-2]
                    f.write(f"{indent}}}\n")

                indent = indent[:-2]
                f.write(f"{indent}}}\n")

            root_idx = [i for i, p in enumerate(self.joint_hierarchy) if p == -1][0]
            write_joint(root_idx, -1)

            f.write("MOTION\n")
            f.write(f"Frames: {root_positions.shape[0]}\n")
            f.write(f"Frame Time: {frame_time:.6f}\n")

            for frame_idx in range(root_positions.shape[0]):
                root_pos = root_positions[frame_idx] / self.scale
                line = [
                    f"{root_pos[0]:.6f}",
                    f"{root_pos[1]:.6f}",
                    f"{root_pos[2]:.6f}"
                ]

                for joint_idx in range(self.n_joints):
                    angles = euler_angles[joint_idx][frame_idx]
                    line.extend([f"{angles[0]:.6f}", f"{angles[1]:.6f}", f"{angles[2]:.6f}"])

                f.write(" ".join(line) + "\n")

    def keypoint_ik(self, root_p, keypoints, roll_offset=None, eps=1e-6):
        # root_p: [b, l, 3]
        # keypoints: [b, l, j, 5], joint tails (global) & roll
        # roll_offset: [b, j]
        # output: [b, l, j, *]

        root_p = torch.clone(root_p, memory_format=torch.contiguous_format)
        keypoints = torch.clone(keypoints, memory_format=torch.contiguous_format)

        directions = keypoints[..., :3]
        rolls = keypoints[..., 3:] # cos, sin order
        rolls = rolls / torch.clamp_min(torch.linalg.vector_norm(rolls, dim=-1, keepdim=True), eps)
        if roll_offset is not None:
            roll_offset = torch.clone(roll_offset, memory_format=torch.contiguous_format)
            roll_offset = (roll_offset * self.roll_mask).unsqueeze(1)

            # complex 2d rotation
            sin_offset = torch.sin(roll_offset)
            cos_offset = torch.cos(roll_offset)
            rolls = torch.stack([cos_offset * rolls[..., 0] - sin_offset * rolls[..., 1],
                                 sin_offset * rolls[..., 0] + cos_offset * rolls[..., 1]], dim=-1)

        heads = []
        q = []
        for i in range(len(self.joint_hierarchy)):
            parent_index = self.joint_hierarchy[i]
            if parent_index == -1:
                head = root_p
            else:
                head = self._qrot(self.joint_offsets[i], q[parent_index])
                head = head + heads[parent_index]

            kp = directions[..., i, :].clone() - head
            q_ = self.batch_single_ik(torch.reshape(self.joint_tails[i], (1, 1, -1)), kp, rolls[..., i, :], eps=eps)

            heads.append(head)
            q.append(q_)

        # heads, tails: [b, l, j, 3], q: [b, l, j, 4]
        heads = torch.stack(heads, dim=2).clone()
        q = torch.stack(q, dim=2).clone()
        return heads, q

    def batch_single_ik(self, raw_u, raw_v, roll, eps=1e-6):
        # u: [..., 3] or [..., l, 3], rest
        # v: [..., l, 3], target
        # roll: [..., l, 2], cos + sin half-angle values

        raw_u = torch.clone(raw_u, memory_format=torch.contiguous_format)
        raw_v = torch.clone(raw_v, memory_format=torch.contiguous_format)
        roll = torch.clone(roll, memory_format=torch.contiguous_format)

        with torch.no_grad():
            raw_u_check = torch.linalg.vector_norm(raw_u, dim=-1)
            raw_u[raw_u_check < eps] = self.id_mat[0, :3]

            raw_v_check = torch.linalg.vector_norm(raw_v, dim=-1)
            raw_v[raw_v_check < eps] = self.id_mat[0, :3]

            roll_check = torch.linalg.vector_norm(roll, dim=-1)
            roll[roll_check < eps] = self.id_mat[0, :2]

        u = raw_u / torch.linalg.vector_norm(raw_u, dim=-1, keepdim=True)
        v = raw_v / torch.linalg.vector_norm(raw_v, dim=-1, keepdim=True)
        if len(raw_u.shape) != len(raw_v.shape):
            u = u.unsqueeze(-2)

        median = (u + v) / 2
        raw_q = torch.cat([torch.full((*median.shape[:-1], 1), eps, device=median.device), median], dim=-1)
        raw_q = raw_q / torch.linalg.vector_norm(raw_q, dim=-1, keepdim=True)

        roll = roll / torch.linalg.vector_norm(roll, dim=-1, keepdim=True)
        cos_roll = roll[..., :1]
        sin_roll = roll[..., 1:]

        roll_q = torch.cat([cos_roll, sin_roll * u], dim=-1)

        q = self._qmul(raw_q, roll_q)
        q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True)

        return q

    def q_to_keypoint(self, q, rest, eps=1e-6):
        # rest: [..., 3]
        # q: [..., l, 4]

        q = q.clone()
        rest = rest.clone()

        with torch.no_grad():
            q_check = torch.linalg.vector_norm(q, dim=-1)
            q[q_check < eps] = self.id_mat

            rest_check = torch.linalg.vector_norm(rest, dim=-1)
            rest[rest_check < eps] = self.id_mat[0, :3]

        u = rest / torch.linalg.vector_norm(rest, dim=-1, keepdim=True)
        u = u.unsqueeze(-2)
        v = self._qrot(u, q)

        median = (u + v) / 2
        raw_q = torch.cat([torch.full((*median.shape[:-1], 1), eps, device=median.device), median], dim=-1)
        raw_q = raw_q / torch.linalg.vector_norm(raw_q, dim=-1, keepdim=True)
        roll_q = self._qmul(-raw_q, q)

        cos_roll = roll_q[..., :1]

        sin_roll = torch.where(torch.abs(u) > eps, roll_q[..., 1:], 0) / (torch.where(torch.abs(u) > eps, u, 1))
        sin_roll_idx = torch.argmax(torch.abs(sin_roll), dim=-1, keepdim=True)
        sin_roll = torch.gather(sin_roll, -1, sin_roll_idx)

        roll = torch.cat([cos_roll, sin_roll], dim=-1)

        return torch.cat([v, roll], dim=-1)

    def _qrot(self, p, q):
        # p: [..., 3], broadcastable
        # q: [..., 4], broadcastable
        if len(p.shape) < len(q.shape):
            shape = [1 for _ in range(len(q.shape) - len(p.shape))]
            shape = [*shape, *p.shape]
            p = torch.reshape(p, shape)
        elif len(q.shape) < len(p.shape):
            shape = [1 for _ in range(len(p.shape) - len(q.shape))]
            shape = [*shape, *q.shape]
            q = torch.reshape(q, shape)

        t = 2 * torch.linalg.cross(q[..., 1:], p, dim=-1)
        new_p = p + (q[..., :1] * t) + torch.linalg.cross(q[..., 1:], t, dim=-1)
        return new_p

    def _qmul(self, u, v):
        # u, v: [..., 4], same dim, real first

        original_shape = u.shape
        terms = torch.bmm(torch.reshape(v, (-1, 4, 1)), torch.reshape(u, (-1, 1, 4)))

        w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
        x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
        y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
        z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
        return torch.reshape(torch.stack((w, x, y, z), dim=1), original_shape).contiguous()




    def generate_pointcloud(self, global_p, global_q, m_lens, n_points=-1, std=-1, output_joints=False):
        # global_p: [b, max_l, j, 3]
        # global_q: [b, max_l, j, 4]
        # m_lens: a list or tensor indicating the actual number of valid frames for each batch (length = b)
        # output: [b, max_l, n_points, 3 + bgroups_dim] (frames beyond m_lens are padded with zeros)

        if n_points == -1:
            n_points = self.n_point
        if std == -1:
            std = self.std

        batches = global_p.shape[0]
        max_len = global_p.shape[1]

        # Sample n_points joint indices from the joint distribution, shape: [n_points]
        joints = self.joint_dist.sample(torch.Size((n_points,))).to(global_p.device)
        # Randomly generate origins for each batch (to control the ratio along the joint's rest position), shape: [b, n_points]
        origins = torch.rand((batches, n_points), dtype=torch.float32, device=global_p.device)
        # Generate normal offsets for each batch for each sampled point, shape: [b, n_points, 3]
        offsets = torch.normal(torch.zeros((batches, n_points, 3), dtype=torch.float32, device=global_p.device), std)

        # Retrieve tail positions and body group data for the sampled joints (assuming self.joint_tails and self.joint_bgroups are defined)
        # tails: [n_points, 3]
        tails = self.joint_tails[joints].clone()
        # bgroups: [n_points, bgroups_dim]
        bgroups = self.joint_bgroups[joints].clone()
        bgroups_dim = bgroups.shape[-1]

        # Initialize the output tensor with zeros, shape: [b, max_l, n_points, 3 + bgroups_dim]
        output_tensor = torch.zeros((batches, max_len, n_points, 3 + bgroups_dim), 
                                    dtype=global_p.dtype, device=global_p.device)

        # Expand tails for broadcasting, shape becomes: [1, n_points, 3]
        tails_exp = tails.unsqueeze(0)

        # Process each batch individually
        for i in range(batches):
            # Get the number of valid frames for the current batch (ensure it is an integer)
            valid_frames = int(m_lens[i].item() if torch.is_tensor(m_lens) else m_lens[i])
            if valid_frames == 0:
                continue

            # Extract the valid frames for the current batch:
            # p_valid: [valid_frames, j, 3] and q_valid: [valid_frames, j, 4]
            p_valid = global_p[i, :valid_frames]
            q_valid = global_q[i, :valid_frames]

            # Construct indices for sampling joints:
            # Reshape joints (shape [n_points]) to [1, n_points, 1] and expand to [valid_frames, n_points, 1]
            indices = joints.view(1, n_points, 1).expand(valid_frames, n_points, 1)

            # Gather joint positions and rotations from the valid frames of the current batch
            # Here, gathering is done along the joint dimension (dim=1)
            heads_sample = torch.gather(p_valid, 1, indices.expand(valid_frames, n_points, 3))
            q_sample = torch.gather(q_valid, 1, indices.expand(valid_frames, n_points, 4))

            # 1. Compute the initial point cloud position along the joint's rest tail position scaled by the origin
            # Reshape origins[i] (shape [n_points]) to [1, n_points, 1] and expand to [valid_frames, n_points, 1]
            origins_i = origins[i].view(1, n_points, 1).expand(valid_frames, n_points, 1)
            initial_pc = tails_exp.expand(valid_frames, n_points, 3) * origins_i

            # 2. Add the normal offset
            offsets_i = offsets[i].unsqueeze(0).expand(valid_frames, n_points, 3)
            pc_with_offset = initial_pc + offsets_i

            # 3. Apply the current global rotation of the joint using the sampled quaternions (q_sample)
            rotated_pc = self._qrot(pc_with_offset, q_sample)

            # 4. Add the joint's global position (head) to get the final global point position
            pc_global = rotated_pc + heads_sample

            # 5. Concatenate the body group data.
            # Expand bgroups from shape [n_points, bgroups_dim] to [valid_frames, n_points, bgroups_dim]
            bgroups_exp = bgroups.unsqueeze(0).expand(valid_frames, n_points, bgroups_dim)
            pc_final = torch.cat([pc_global, bgroups_exp], dim=-1)

            # Assign the computed point cloud for the valid frames into the output tensor; the remaining frames stay zero (padding)
            output_tensor[i, :valid_frames] = pc_final

        if output_joints:
            return output_tensor, joints
        else:
            return output_tensor


    def get_tails(self, global_p, global_q):
        # global_p & global_q: [..., j, 4]
        tail_offset = self._qrot(self.joint_tails, global_q)
        tails = global_p + tail_offset

        return tails