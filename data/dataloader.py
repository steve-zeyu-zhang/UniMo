import random

import torch



class DataLoader:
    def __init__(self, data_folder, skeleton_file, min_length=0, step=1, data_device='cpu', skel_device='cpu'):
        self.data_folder = data_folder

        raw_data = self.load_files()

        # 移除多进程池，改用单进程
        motion_data = [self.process_data(d) for d in raw_data]  # 直接在主进程处理

        self.min_length = min_length
        self.motion_data = []
        for data in motion_data:
            if data.shape[0] // step >= min_length:
                self.motion_data.append(data.to(data_device))

        self.skeleton = self.setup_skeleton(skeleton_file, skel_device)

    def load_files(self):
        pass

    def process_data(self, data):
        pass

    @staticmethod
    def setup_skeleton(skeleton_file, device):
        pass

    def load_samples(self, n, length):
        samples = []

        for i in range(n):
            sample_idx = random.randrange(0, len(self.motion_data))
            start = random.randint(0, self.motion_data[sample_idx].shape[0] - length)
            samples.append(self.motion_data[sample_idx][start:start + length])

        samples = torch.stack(samples, dim=0)
        return samples[..., :3].clone(), torch.reshape(samples[..., 3:].clone(), (*samples.shape[:2], -1, 4))
