import torch
import librosa
import os
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, ClapTextModelWithProjection
from src.models.transformer import Dasheng_Encoder
from src.models.sed_decoder import Decoder, TSED_Wrapper
from src.utils import load_yaml_with_includes
from tqdm import tqdm

class FlexSED:
    def __init__(
        self,
        config_path='src/configs/model.yml',
        ckpt_path='ckpts/flexsed_as.pt',
        ckpt_url='https://huggingface.co/Higobeatz/FlexSED/resolve/main/ckpts/flexsed_as.pt',
        device='cuda'
    ):
        """
        Initialize FlexSED with model, CLAP, and tokenizer loaded once.
        If the checkpoint is not available locally, it will be downloaded automatically.
        """
        self.device = device
        params = load_yaml_with_includes(config_path)

        # Ensure checkpoint exists
        if not os.path.exists(ckpt_path):
            print(f"[FlexSED] Downloading checkpoint from {ckpt_url} ...")
            state_dict = torch.hub.load_state_dict_from_url(ckpt_url, map_location="cpu")
        else:
            state_dict = torch.load(ckpt_path, map_location="cpu")

        # Encoder + Decoder
        encoder = Dasheng_Encoder(**params['encoder']).to(self.device)
        decoder = Decoder(**params['decoder']).to(self.device)
        self.model = TSED_Wrapper(encoder, decoder, params['ft_blocks'], params['frozen_encoder'])
        self.model.load_state_dict(state_dict['model'])
        self.model.eval()

        # CLAP text model
        self.clap = ClapTextModelWithProjection.from_pretrained("laion/clap-htsat-unfused")
        self.clap.eval()
        self.tokenizer = AutoTokenizer.from_pretrained("laion/clap-htsat-unfused")

    def split_audio_fixed(self, audio, sr, chunk_duration=10.0):
        samples_per_chunk = int(sr * chunk_duration)
        total_len = len(audio)

        chunks = []
        for start in range(0, total_len, samples_per_chunk):
            end = min(start + samples_per_chunk, total_len)
            chunks.append(audio[start:end])
        return chunks

    def run_inference(self, audio_path, events, norm_audio=True):
        """
        Run inference on audio for given events.
        """

        # Get CLAP embeddings for each event
        clap_embeds = []
        with torch.no_grad():
            for event in events:
                text = f"The sound of {event.replace('_', ' ').capitalize()}"
                inputs = self.tokenizer([text], padding=True, return_tensors="pt")
                outputs = self.clap(**inputs)
                text_embeds = outputs.text_embeds.unsqueeze(1)
                clap_embeds.append(text_embeds)

            query = torch.cat(clap_embeds, dim=1).to(self.device)

        audio, sr = librosa.load(audio_path, sr=16000)
        # Chunk audio into 10s segments
        audio_chunks = self.split_audio_fixed(audio, sr, 10)
        # Run inference on each chunk
        preds_list = []
        for chunk in tqdm(audio_chunks):
            chunk = torch.tensor([chunk]).to(self.device)

            if norm_audio:
                eps = 1e-9
                max_val = torch.max(torch.abs(chunk))
                chunk = chunk / (max_val + eps)

            mel = self.model.forward_to_spec(chunk)
            preds = self.model(mel, query)
            preds = torch.sigmoid(preds).cpu()
            preds_list.append(preds)

        # Concatenate predictions
        preds = torch.cat(preds_list, dim=2)

        return preds  # shape: [num_events, 1, T]

    # ---------- Multi-event plotting ----------
    @staticmethod
    def plot_and_save_multi(preds, events, sr=25, out_dir="./plots", fname="all_events"):
        os.makedirs(out_dir, exist_ok=True)
        preds_np = preds.squeeze(1).detach().numpy()  # [num_events, T]
        T = preds_np.shape[1]

        plt.figure(figsize=(12, len(events) * 0.6 + 2))
        plt.imshow(
            preds_np,
            aspect="auto",
            cmap="Blues",
            extent=[0, T/sr, 0, len(events)],
            vmin=0, vmax=1, origin="lower"

        )
        plt.colorbar(label="Probability")
        plt.yticks(np.arange(len(events)) + 0.5, events)
        plt.xlabel("Time (s)")
        plt.ylabel("Events")
        plt.title("Event Predictions")

        save_path = os.path.join(out_dir, f"{fname}.png")
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
        return save_path

    def to_multi_plot(self, preds, events, out_dir="./plots", fname="all_events"):
        return self.plot_and_save_multi(preds, events, out_dir=out_dir, fname=fname)

    # ---------- Multi-event video ----------
    @staticmethod
    def make_multi_event_video(preds, events, sr=25, out_dir="./videos",
                               audio_path=None, fps=25, highlight=True, fname="all_events"):
        from moviepy.editor import ImageSequenceClip, AudioFileClip
        from tqdm import tqdm

        os.makedirs(out_dir, exist_ok=True)
        preds_np = preds.squeeze(1).numpy()  # [num_events, T]
        T = preds_np.shape[1]
        duration = T / sr

        frames = []
        n_frames = int(duration * fps)

        for i in tqdm(range(n_frames)):
            t = int(i * T / n_frames)
            plt.figure(figsize=(12, len(events) * 0.6 + 2))

            if highlight:
                mask = np.zeros_like(preds_np)
                mask[:, :t+1] = preds_np[:, :t+1]
                plt.imshow(
                    mask,
                    aspect="auto",
                    cmap="Blues",
                    extent=[0, T/sr, 0, len(events)],
                    vmin=0, vmax=1, origin="lower"
                )
            else:
                plt.imshow(
                    preds_np[:, :t+1],
                    aspect="auto",
                    cmap="Blues",
                    extent=[0, (t+1)/sr, 0, len(events)],
                    vmin=0, vmax=1, origin="lower"
                )

            plt.colorbar(label="Probability")
            plt.yticks(np.arange(len(events)) + 0.5, events)
            plt.xlabel("Time (s)")
            plt.ylabel("Events")
            plt.title("Event Predictions")

            frame_path = f"/tmp/frame_{i:04d}.png"
            plt.savefig(frame_path, dpi=150, bbox_inches="tight")
            plt.close()
            frames.append(frame_path)

        clip = ImageSequenceClip(frames, fps=fps)
        if audio_path is not None:
            audio = AudioFileClip(audio_path).subclip(0, duration)
            clip = clip.set_audio(audio)

        save_path = os.path.join(out_dir, f"{fname}.mp4")
        clip.write_videofile(
            save_path,
            fps=fps,
            codec="mpeg4",
            audio_codec="aac"
        )

        for f in frames:
            os.remove(f)

        return save_path

    def to_multi_video(self, preds, events, audio_path, out_dir="./videos", fname="all_events"):
        return self.make_multi_event_video(
            preds, events, audio_path=audio_path, out_dir=out_dir, fname=fname
        )


if __name__ == "__main__":
    flexsed = FlexSED(device='cuda')

    events = ["Door", "Male Speech", "Laughter", "Dog"]
    preds = flexsed.run_inference("example.wav", events)

    # Combined plot & video
    flexsed.to_multi_plot(preds, events, fname="example")
    # flexsed.to_multi_video(preds, events, audio_path="example.wav", fname="example")
