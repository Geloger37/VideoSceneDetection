import Model
import Dataset
import argparse
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import numpy as np
from typing import Tuple
import json
import pandas as pd
import os


def train(model: nn.Module, dataloader: DataLoader, epochs: int = 10):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        total_loss = 0.0
        for batch in dataloader:
            video_frames = batch["video_frames"].cuda()
            audio_waveform = batch["audio_waveform"].cuda()
            labels = batch["labels"].cuda()

            optimizer.zero_grad()
            outputs = model(video_frames, audio_waveform)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")


def predict_intro(
    model: nn.Module, video_path: str, threshold: float = 0.5
) -> Tuple[int, int]:
    model.eval()

    # Извлечение данных
    dataset = Dataset.VideoAudioDataset(os.path.dirname(video_path), {})
    sample = dataset._extract_frames(video_path), dataset._load_audio(video_path)

    with torch.no_grad():
        outputs = model(
            torch.FloatTensor(sample["video_frames"]).unsqueeze(0).cuda(),
            torch.FloatTensor(sample["audio_waveform"]).unsqueeze(0).cuda(),
        )
        probs = outputs.squeeze().cpu().numpy()

    # Поиск заставки
    intro_start = np.argmax(probs > threshold)
    intro_end = len(probs) - np.argmax(probs[::-1] > threshold)
    return intro_start, intro_end


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainFolder", type=str, default="data_train_short")
    parser.add_argument("--testFolder", type=str, default="data_test_short")
    parser.add_argument(
        "--pretrainedAudioModel", type=str, default="facebook/wav2vec2-base-960h"
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    with open("data.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    json_df = pd.DataFrame.from_dict(data, orient="index")
    json_data = {}
    for row in json_df.iterrows():
        videoName = row["url"].split("-")[-1] + ".mp4"
        json_data[videoName] = (row[start], row[end])

    dataset = Dataset.VideoAudioDataset(
        args.pretrainedAudioModel, args.trainFolder, json_data
    )
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Модель и обучение
    model = Model.AVAttentionModel(
        args.pretrainedAudioModel
    ).cuda()  # TODO: Поменять на динамический выбор
    train(model, dataloader, epochs=10)

    # Тестирование
    video_path = os.path.join(
        args.testFolder, "-220020068_456241671", "-220020068_456241671.mp4"
    )
    start, end = predict_intro(model, video_path)
    print(f"Заставка: {start}-{end} сек")


if __name__ == "__main__":
    main()
