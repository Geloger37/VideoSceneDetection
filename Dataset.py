import numpy as np
from torch.utils.data import Dataset
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import torch
from typing import Dict, Tuple
import librosa
import cv2
import os


class VideoAudioDataset(Dataset):
    def __init__(
        self,
        pretrainedAudioModel: str,
        data_dir: str,
        json_data: Dict[str, Tuple[str, str]],
    ):
        self.data_dir = data_dir
        self.video_files = [f for f in os.listdir(data_dir) if f.endswith(".mp4")]
        self.json_data = json_data

        # Инициализация Wav2Vec
        self.audio_processor = Wav2Vec2Processor.from_pretrained(pretrainedAudioModel)
        self.audio_model = (
            Wav2Vec2Model.from_pretrained(pretrainedAudioModel).eval().cuda()
        )  # TODO: Нужно поменять на динамический вариант

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = os.path.join(self.data_dir, self.video_files[idx])

        # Извлечение кадров (1 кадр/сек)
        frames = self._extract_frames(video_path)

        # Извлечение аудио (сырой waveform для Wav2Vec)
        waveform = self._load_audio(video_path)

        # Разметка (0/1 для каждого кадра)
        start, end = self.json_data[self.video_files[idx]]
        labels = self._generate_labels(frames.shape[0], start, end)

        return {
            "video_frames": torch.FloatTensor(frames),
            "audio_waveform": torch.FloatTensor(waveform),
            "labels": torch.FloatTensor(labels),
        }

    def _extract_frames(self, video_path: str, fps: int = 1) -> np.ndarray:
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (224, 224))
            frame = frame.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)
            frames.append(frame)
        cap.release()
        return np.stack(frames[::fps])  # (N_frames, 3, 224, 224)

    def _load_audio(self, video_path: str) -> np.ndarray:
        # Извлечение аудио из видео (16 kHz, моно)
        audio_path = video_path.replace(".mp4", ".wav")
        os.system(f"ffmpeg -i {video_path} -ac 1 -ar 16000 {audio_path} -y")
        waveform, _ = librosa.load(audio_path, sr=16000)
        return waveform  # (T,)

    def _generate_labels(self, n_frames: int, start: str, end: str) -> np.ndarray:
        start_sec = sum(x * int(t) for x, t in zip([3600, 60, 1], start.split(":")))
        end_sec = sum(x * int(t) for x, t in zip([3600, 60, 1], end.split(":")))
        labels = np.zeros(n_frames)
        labels[start_sec:end_sec] = 1  # 1 кадр/сек
        return labels
