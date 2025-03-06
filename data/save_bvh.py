import torch
import numpy as np

class BVHExporter:
    def __init__(self, joint_names, joint_hierarchy, joint_offsets, joint_tails):
        self.joint_names = joint_names
        self.joint_hierarchy = joint_hierarchy
        self.joint_offsets = joint_offsets
        self.joint_tails = joint_tails
        self.n_joints = len(joint_names)
        self.scale = 0.01  # 根据实际数据尺度调整

    def save_as_bvh(self, global_p, global_q, filename, frame_time=1.0/30.0):
        """改进的BVH导出方法，确保正确的人体结构"""
        # 转换为CPU numpy数组
        global_p = global_p.detach().cpu().numpy()
        global_q = global_q.detach().cpu().numpy()

        with open(filename, 'w') as f:
            # ================== HIERARCHY部分 ==================
            f.write("HIERARCHY\n")

            def write_joint(joint_idx, indent=0):
                indent_str = '  ' * indent
                joint_name = self.joint_names[joint_idx]

                # 节点声明
                if self.joint_hierarchy[joint_idx] == -1:
                    f.write(f"{indent_str}ROOT {joint_name}\n")
                else:
                    f.write(f"{indent_str}JOINT {joint_name}\n")

                f.write(f"{indent_str}{{\n")
                indent += 1
                indent_str = '  ' * indent

                # 偏移量处理（Y-up坐标系）
                offset = self.joint_offsets[joint_idx].detach().cpu().numpy()
                converted_offset = self._convert_coordinates(offset / self.scale)
                f.write(f"{indent_str}OFFSET {converted_offset[0]:.6f} {converted_offset[1]:.6f} {converted_offset[2]:.6f}\n")

                # 通道声明
                if self.joint_hierarchy[joint_idx] == -1:
                    f.write(f"{indent_str}CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation\n")
                else:
                    f.write(f"{indent_str}CHANNELS 3 Zrotation Xrotation Yrotation\n")

                # 递归处理子节点
                children = [i for i, p in enumerate(self.joint_hierarchy) if p == joint_idx]
                for child in children:
                    write_joint(child, indent)

                # 末端节点处理
                if not children:
                    f.write(f"{indent_str}End Site\n")
                    f.write(f"{indent_str}{{\n")
                    indent += 1
                    indent_str = '  ' * indent

                    tail = self.joint_tails[joint_idx].detach().cpu().numpy()
                    converted_tail = self._convert_coordinates(tail / self.scale)
                    f.write(f"{indent_str}OFFSET {converted_tail[0]:.6f} {converted_tail[1]:.6f} {converted_tail[2]:.6f}\n")

                    indent -= 1
                    f.write(f"{'  ' * indent}}}\n")

                indent -= 1
                f.write(f"{'  ' * indent}}}\n")

            # 从根节点开始写入
            root_idx = np.where(np.array(self.joint_hierarchy) == -1)[0][0]
            write_joint(root_idx)

            # ================== MOTION部分 ==================
            f.write("MOTION\n")
            f.write(f"Frames: {global_p.shape[0]}\n")
            f.write(f"Frame Time: {frame_time:.6f}\n")

            for frame_idx in range(global_p.shape[0]):
                frame_data = []

                # 根节点处理
                root_pos = self._convert_coordinates(global_p[frame_idx, 0] / self.scale)
                frame_data.extend(root_pos)

                # 根节点旋转（ZXY顺序）
                root_rot = self._quaternion_to_euler(global_q[frame_idx, 0], "ZXY")
                root_rot_deg = np.degrees(root_rot)
                frame_data.extend([root_rot_deg[2], root_rot_deg[0], root_rot_deg[1]])

                # 其他关节处理
                for joint_idx in range(1, self.n_joints):
                    parent_idx = self.joint_hierarchy[joint_idx]
                    local_rot = self._get_local_rotation(
                        global_q[frame_idx, parent_idx],
                        global_q[frame_idx, joint_idx]
                    )
                    euler = self._quaternion_to_euler(local_rot, "ZXY")
                    euler_deg = np.degrees(euler)
                    frame_data.extend([euler_deg[2], euler_deg[0], euler_deg[1]])

                f.write(" ".join(f"{x:.6f}" for x in frame_data) + "\n")

    def _convert_coordinates(self, vec):
        """坐标系转换：XZY->XYZ with Y-up"""
        return [vec[0], -vec[2], vec[1]]

    def _quaternion_to_euler(self, q, order):
        """四元数转欧拉角（ZXY顺序）"""
        matrix = self.quaternion_to_rotation_matrix(torch.tensor(q).unsqueeze(0))
        euler = self._matrix_to_euler(matrix, order)
        euler = euler.numpy()[0]

        # 角度规范化
        euler_deg = np.degrees(euler)
        euler_deg = (euler_deg + 180) % 360 - 180
        return np.radians(euler_deg)

    @staticmethod
    def quaternion_to_rotation_matrix(q):
        """四元数转旋转矩阵（支持批量处理）"""
        q = q / torch.norm(q, dim=-1, keepdim=True)
        w, x, y, z = q.unbind(-1)
        return torch.stack([
            1-2*y*y-2*z*z,   2*x*y-2*z*w,   2*x*z+2*y*w,
            2*x*y+2*z*w,   1-2*x*x-2*z*z,   2*y*z-2*x*w,
            2*x*z-2*y*w,   2*y*z+2*x*w,   1-2*x*x-2*y*y
        ], dim=-1).view(*q.shape[:-1], 3, 3)

    def _matrix_to_euler(self, matrix, order):
        """矩阵转欧拉角（ZXY顺序）"""
        if order == "ZXY":
            z = torch.atan2(matrix[..., 0, 1], matrix[..., 1, 1])
            x = torch.atan2(-matrix[..., 2, 1],
                          torch.sqrt(matrix[..., 2, 0]**2 + matrix[..., 2, 2]**2 + 1e-10))
            y = torch.atan2(matrix[..., 2, 0], matrix[..., 2, 2])
            return torch.stack((z, x, y), dim=-1)
        else:
            raise ValueError(f"Unsupported Euler order: {order}")

    def _get_local_rotation(self, parent_q, joint_q):
        """计算本地旋转（parent_q需要是单位四元数）"""
        parent_conj = np.array([parent_q[0], -parent_q[1], -parent_q[2], -parent_q[3]])
        return self._quaternion_multiply(parent_conj, joint_q)

    def _quaternion_multiply(self, q1, q2):
        """四元数乘法（顺序修正）"""
        return np.array([
            q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3],
            q1[0]*q2[1] + q1[1]*q2[0] + q1[2]*q2[3] - q1[3]*q2[2],
            q1[0]*q2[2] - q1[1]*q2[3] + q1[2]*q2[0] + q1[3]*q2[1],
            q1[0]*q2[3] + q1[1]*q2[2] - q1[2]*q2[1] + q1[3]*q2[0]
        ])