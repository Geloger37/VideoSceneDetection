import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor
from torchvision import models


class AVAttentionModel(nn.Module):
    def __init__(
        self,
        pretrainedAudioModel: str,
        audio_feat_dim: int = 768,
        video_feat_dim: int = 2048,
        hidden_dim: int = 256,
    ):
        super().__init__()

        # Видеоэнкодер (ResNet50)
        self.video_encoder = models.resnet50(pretrained=True)
        self.video_encoder.fc = nn.Identity()
        self.video_proj = nn.Linear(video_feat_dim, hidden_dim)

        # Аудиоэнкодер (Wav2Vec 2.0)
        self.audio_proj = nn.Linear(audio_feat_dim, hidden_dim)

        self.audio_processor = Wav2Vec2Processor.from_pretrained(pretrainedAudioModel)

        # Механизмы внимания
        self.cross_attention = nn.MultiheadAttention(hidden_dim, num_heads=4)

        # Классификатор
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128), nn.ReLU(), nn.Linear(128, 1), nn.Sigmoid()
        )

    def forward(
        self, video_frames: torch.Tensor, audio_waveform: torch.Tensor
    ) -> torch.Tensor:
        # Видеопризнаки (N_frames, 2048)
        video_feats = self.video_encoder(video_frames)
        video_feats = self.video_proj(video_feats)  # (N_frames, hidden_dim)

        # Аудиопризнаки (T_audio, 768)
        with torch.no_grad():
            inputs = self.audio_processor(
                audio_waveform.numpy(),
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
            ).to("cuda")
            audio_feats = self.audio_model(
                **inputs
            ).last_hidden_state  # (1, T_audio, 768)
        audio_feats = self.audio_proj(audio_feats.squeeze(0))  # (T_audio, hidden_dim)

        # Cross-Modal Attention
        attended_video, _ = self.cross_attention(
            query=video_feats.unsqueeze(1),  # (N_frames, 1, hidden_dim)
            key=audio_feats.unsqueeze(1),
            value=audio_feats.unsqueeze(1),
        )
        attended_video = attended_video.squeeze(1)

        attended_audio, _ = self.cross_attention(
            query=audio_feats.unsqueeze(1),  # (T_audio, 1, hidden_dim)
            key=video_feats.unsqueeze(1),
            value=video_feats.unsqueeze(1),
        )
        attended_audio = attended_audio.squeeze(1)

        # Усреднение
        video_global = attended_video.mean(dim=0)  # (hidden_dim)
        audio_global = attended_audio.mean(dim=0)  # (hidden_dim)

        # Классификация
        combined = torch.cat([video_global, audio_global], dim=0)
        return self.classifier(combined.unsqueeze(0))
