import glob
import os
import random
from pdb import set_trace

import numpy as np
import resampy
import torch
from audlib.io.audio import audioinfo, audioread, audiowrite
from pytorch_lightning import LightningDataModule
from scipy.io import loadmat
from scipy.signal import convolve
from torch.utils.data import DataLoader, Dataset
from torchaudio.transforms import Resample


def build_loaders(params):
    """
    Builds dataloaders for baseline experiments
    """
    data_train = DoaData(params, "train")
    data_val = DoaData(params, "val")
    data_test = DoaData(params, "test")

    train_loader = DataLoader(
        dataset=data_train,
        batch_size=params.batch_size,
        num_workers=params.nworkers,
        shuffle=True,
    )
    val_loader = DataLoader(
        dataset=data_val,
        batch_size=params.batch_size,
        num_workers=params.nworkers,
        shuffle=False,
    )
    test_loader = DataLoader(
        dataset=data_test,
        batch_size=params.batch_size,
        num_workers=params.nworkers,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader


class DoaDataModule(LightningDataModule):
    def __init__(self, params):
        self.params = params
        super().__init__()

    def setup(self, stage):
        if self.params.noise_type == "noisy":
            self.data_train = NoisyDoaData(self.params, "train")
            self.data_val = NoisyDoaData(self.params, "val")
            self.data_test = NoisyDoaData(self.params, "test")
        elif self.params.noise_type == "val_noisy":
            self.data_train = DoaData(self.params, "train")
            self.data_val = NoisyDoaData(self.params, "val")
            self.data_test = NoisyDoaData(self.params, "test")
        else:
            self.data_train = DoaData(self.params, "train")
            self.data_val = DoaData(self.params, "val")
            self.data_test = DoaData(self.params, "test")

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.params.batch_size,
            num_workers=self.params.nworkers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.params.batch_size,
            num_workers=self.params.nworkers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.params.batch_size,
            num_workers=self.params.nworkers,
            shuffle=False,
        )


class DoaData(Dataset):
    def __init__(self, params, set_type="train"):
        super().__init__()
        if params.sample_rate == 16000:
            suffix = "16k"
        elif params.sample_rate == 48000:
            suffix = ""
        elif params.sample_rate == 11025:
            suffix = "11k"

        if "vbd" in params.speech_root:
            if set_type == "train":
                speech_dir = os.path.join(
                    params.speech_root, "clean_trainset_28spk_wav{}".format(suffix)
                )
            elif set_type == "test":
                speech_dir = os.path.join(
                    params.speech_root, "new_clean_testset_wav{}".format(suffix)
                )
            elif set_type == "val":
                speech_dir = os.path.join(
                    params.speech_root, "new_clean_valset_wav{}".format(suffix)
                )
            else:
                exit()
            self.speech_files = glob.glob(os.path.join(speech_dir, "*.wav"))
        elif "LibriSpeech" in params.speech_root:
            if set_type == "train":
                speech_dir = os.path.join(
                    params.speech_root, "train-clean-100-{}".format(suffix)
                )
            else:
                speech_dir = os.path.join(
                    params.speech_root, "dev-clean-{}".format(suffix)
                )
            self.speech_files = glob.glob(os.path.join(speech_dir, "*", "*", "*.flac"))

        if "wav" in params.rir_root:
            rir_dir = params.rir_root
            if set_type == "train":
                self.rir_files = glob.glob(
                    os.path.join(rir_dir, "train", "left", "*.wav")
                )
                self.fixed_duration = params.fixed_duration
            else:
                self.rir_files = glob.glob(
                    os.path.join(rir_dir, "val", "left", "*.wav")
                )
                # self.fixed_duration = None
                self.fixed_duration = params.fixed_duration

            self.rir_filetype = "wav"
            self.angles = []
            for filename in self.rir_files:
                angle = filename.split("_")[-1].split(".")[0]
                if "neg" in angle:
                    sign = -1
                    angle = angle.replace("neg", "")
                else:
                    sign = 1
                self.angles.append(sign * int(angle[:-2]) * np.pi / 180)
        elif "matlab" in params.rir_root:
            if set_type == "train":
                rir_dir = os.path.join(params.rir_root, "train")
                self.fixed_duration = params.fixed_duration
            else:
                rir_dir = os.path.join(params.rir_root, "val")
                self.fixed_duration = None

            self.rir_filetype = "mat"
            self.rir_files = glob.glob(os.path.join(rir_dir, "*.mat"))
            self.angles = np.genfromtxt(
                os.path.join(rir_dir, "annotations.txt"), delimiter=",", skip_header=1
            )[:, -2]
        else:
            if set_type == "train":
                rir_dir = os.path.join(params.rir_root, "train")
                self.fixed_duration = params.fixed_duration
            else:
                rir_dir = os.path.join(params.rir_root, "val")
                # self.fixed_duration = None
                self.fixed_duration = params.fixed_duration

            self.rir_filetype = "npy"
            self.rir_files = glob.glob(os.path.join(rir_dir, "*.npy"))
            self.angles = np.genfromtxt(
                os.path.join(rir_dir, "annotations.txt"), delimiter=",", skip_header=1
            )[:, -2]
        self.train = set_type == "train"
        self.sample_rate = params.sample_rate
        self.middle = params.middle
        self.set_len = len(self.speech_files)
        self.save = params.debug
        self.simple = params.simple_input
        self.n_angles = params.n_angles
        self.resample = params.resample

        if self.train:
            self.n_angles += 2
        self.binary = params.binary

    def __getitem__(self, idx):
        out = {}
        file_path = self.speech_files[idx]
        if self.train:
            rir_idx = random.randrange(len(self.rir_files))
        else:
            rir_idx = idx % len(self.rir_files)
        rir_path = self.rir_files[rir_idx]
        speech = self.read(file_path)

        if self.resample is not None:
            speech = resampy.resample(speech, self.sample_rate, self.resample)

        if self.simple:
            speech_up = resampy.resample(speech, 1, self.n_angles)
            angle_idx = int(
                np.random.choice(np.arange(self.n_angles)) - self.n_angles // 2
            )
            if angle_idx < 0:
                out["left"] = speech
                speech_shift = np.hstack((np.zeros(np.abs(angle_idx)), speech_up[:-1]))
                out["right"] = resampy.resample(speech_shift, self.n_angles, 1)
            elif angle_idx > 0:
                out["right"] = speech
                speech_shift = np.hstack((np.zeros(np.abs(angle_idx)), speech_up[:-1]))
                out["left"] = resampy.resample(speech_shift, self.n_angles, 1)
            else:
                out["right"] = speech
                out["left"] = speech
            lags = angle_idx / self.n_angles
            y = np.arcsin(343 * lags / 0.02 / self.sample_rate)
        else:
            if self.rir_filetype == "npy":
                rir = np.load(rir_path)
                out["left"] = convolve(speech, rir[0])[: len(speech)]
                out["right"] = convolve(speech, rir[2])[: len(speech)]
                y = self.angles[int(rir_path.split("_")[-1].split(".")[0])]
            elif self.rir_filetype == "mat":
                rir = loadmat(rir_path)["rir_mat"]
                out["left"] = convolve(speech, rir[0])[: len(speech)]
                out["right"] = convolve(speech, rir[2])[: len(speech)]
                y = self.angles[int(rir_path.split("_")[-1].split(".")[0])]
            elif self.rir_filetype == "wav":
                rir_left, _ = audioread(rir_path)
                rir_right, _ = audioread(rir_path.replace("left", "right"))
                out["left"] = convolve(speech, rir_left)[: len(speech)]
                out["right"] = convolve(speech, rir_right)[: len(speech)]
                y = self.angles[rir_idx]

        norm_factor = np.maximum(
            np.max(np.abs(out["left"])), np.max(np.abs(out["right"]))
        )
        out["left"] /= norm_factor
        out["right"] /= norm_factor

        if self.save:
            print(y, rir_path, file_path)
            audiowrite("sample.wav", speech, self.sample_rate)
            audiowrite(
                "sample_sim.wav",
                np.vstack((out["left"], out["right"])).T,
                self.sample_rate,
            )

        if self.binary:
            y = 1 if y > 0 else 0
        return out['left'].astype('float32'), out['right'].astype('float32'), y.astype('float32')

    def __len__(self):
        return self.set_len

    def read(self, path):
        """Read in audio samples from a list of files."""
        info = audioinfo(path)
        assert info.samplerate == self.sample_rate
        if self.fixed_duration is None:
            signal, _ = audioread(path)
            return signal

        duration = int(info.samplerate * self.fixed_duration)
        if info.frames < duration:  # pad
            if self.middle:
                start = int((duration - info.frames) / 2)
            else:
                start = self.random.randrange(duration - info.frames)
            signal = np.zeros(duration)
            tmp, _ = audioread(path)
            signal[start : start + len(tmp)] = tmp
        else:
            if self.middle:
                start = int((info.frames - duration + 1) / 2)
            else:
                start = self.random.randrange(info.frames - duration + 1)
            stop = start + duration
            signal, _ = audioread(path, start=start, stop=stop)

        return signal


class NoisyDoaData(Dataset):
    def __init__(self, params, set_type="train"):
        super().__init__()
        if params.sample_rate == 16000:
            suffix = "16k"
        elif params.sample_rate == 48000:
            suffix = ""
        elif params.sample_rate == 11025:
            suffix = "11k"

        if "vbd" in params.speech_root:
            if set_type == "train":
                speech_dir = os.path.join(
                    params.speech_root, "clean_trainset_28spk_wav{}".format(suffix)
                )
            elif set_type == "test":
                speech_dir = os.path.join(
                    params.speech_root, "new_clean_testset_wav{}".format(suffix)
                )
            elif set_type == "val":
                speech_dir = os.path.join(
                    params.speech_root, "new_clean_valset_wav{}".format(suffix)
                )
            else:
                exit()
            self.speech_files = glob.glob(os.path.join(speech_dir, "*.wav"))
            self.additive = True
        elif "LibriSpeech" in params.speech_root:
            if set_type == "train":
                speech_dir = os.path.join(
                    params.speech_root, "train-clean-100-{}".format(suffix)
                )
            else:
                speech_dir = os.path.join(
                    params.speech_root, "dev-clean-{}".format(suffix)
                )
            self.speech_files = glob.glob(os.path.join(speech_dir, "*", "*", "*.flac"))
            self.additive = False

        if "wav" in params.rir_root:
            rir_dir = params.rir_root
            if set_type == "train":
                self.rir_files = glob.glob(
                    os.path.join(rir_dir, "train", "left", "*.wav")
                )
                self.fixed_duration = params.fixed_duration
            else:
                self.rir_files = glob.glob(
                    os.path.join(rir_dir, "val", "left", "*.wav")
                )
                self.fixed_duration = None

            self.rir_filetype = "wav"
            self.angles = []
            for filename in self.rir_files:
                angle = filename.split("_")[-1].split(".")[0]
                if "neg" in angle:
                    sign = -1
                    angle = angle.replace("neg", "")
                else:
                    sign = 1
                self.angles.append(sign * int(angle[:-2]) * np.pi / 180)
        elif "matlab" in params.rir_root:
            if set_type == "train":
                rir_dir = os.path.join(params.rir_root, "train")
                self.fixed_duration = params.fixed_duration
            else:
                rir_dir = os.path.join(params.rir_root, "val")
                self.fixed_duration = None

            self.rir_filetype = "mat"
            self.rir_files = glob.glob(os.path.join(rir_dir, "*.mat"))
            self.angles = np.genfromtxt(
                os.path.join(rir_dir, "annotations.txt"), delimiter=",", skip_header=1
            )[:, -2]
        else:
            if set_type == "train":
                rir_dir = os.path.join(params.rir_root, "train")
                self.fixed_duration = params.fixed_duration
            else:
                rir_dir = os.path.join(params.rir_root, "val")
                self.fixed_duration = None

            self.rir_filetype = "npy"
            self.rir_files = glob.glob(os.path.join(rir_dir, "*.npy"))
            self.angles = np.genfromtxt(
                os.path.join(rir_dir, "annotations.txt"), delimiter=",", skip_header=1
            )[:, -2]
        self.train = set_type == "train"
        self.sample_rate = params.sample_rate
        self.middle = params.middle
        if params.n_train_samples:
            self.set_len = params.n_train_samples
        else:
            self.set_len = len(self.speech_files)
        self.save = params.debug
        self.simple = params.simple_input
        self.n_angles = params.n_angles
        if set_type == "train":
            self.n_angles += 2
        self.binary = params.binary
        self.white_noise = params.white_noise
        self.snr = params.snr
        self.noise_prob = params.noise_prob

    def __getitem__(self, idx):
        out = {}
        file_path = self.speech_files[idx]
        noise_path = file_path.replace("clean", "noisy")
        if self.train:
            rir_idx = random.randrange(len(self.rir_files))
        else:
            rir_idx = idx % len(self.rir_files)
        rir_path = self.rir_files[rir_idx]
        speech, noise = self.read(file_path, noise_path)
        if self.snr is None:
            # Mimic VBD SNR
            sig_power = np.sum(speech**2)
            noise_power = np.sum(noise**2)
            snr = 10 * np.log10(sig_power / noise_power) + 10

            # Random SNR in range
            # snr = np.random.uniform(0,20)
        else:
            snr = self.snr
        if self.white_noise:
            if random.uniform(0, 1) >= 0.5 or not self.train:
                noise = np.random.normal(0, 1, len(speech))
                speech_power = np.sum(speech**2)
                noise_power = np.sum(noise**2)
                power_norm = np.sqrt(10 ** (snr / 10) * noise_power / speech_power)
                noise = noise / power_norm
            else:
                noise = np.zeros(len(speech))

        if self.simple:
            speech_up = resampy.resample(speech, 1, self.n_angles)
            angle_idx = int(
                np.random.choice(np.arange(self.n_angles)) - self.n_angles // 2
            )
            if angle_idx < 0:
                out["left"] = speech
                speech_shift = np.hstack((np.zeros(np.abs(angle_idx)), speech_up[:-1]))
                out["right"] = resampy.resample(speech_shift, self.n_angles, 1)
            elif angle_idx > 0:
                out["right"] = speech
                speech_shift = np.hstack((np.zeros(np.abs(angle_idx)), speech_up[:-1]))
                out["left"] = resampy.resample(speech_shift, self.n_angles, 1)
            else:
                out["right"] = speech
                out["left"] = speech
            lags = angle_idx / self.n_angles
            y = np.arcsin(343 * lags / 0.02 / self.sample_rate)
        else:
            if self.rir_filetype == "npy":
                rir = np.load(rir_path)
                out["left"] = convolve(speech, rir[0])[: len(speech)]
                out["right"] = convolve(speech, rir[2])[: len(speech)]
                if np.random.rand() < self.noise_prob or not self.train:
                    out["left"] += convolve(noise, rir[1])[: len(speech)]
                    out["right"] += convolve(noise, rir[3])[: len(speech)]
                y = self.angles[int(rir_path.split("_")[-1].split(".")[0])]
            elif self.rir_filetype == "mat":
                rir = loadmat(rir_path)["rir_mat"]
                out["left"] = convolve(speech, rir[0])[: len(speech)]
                out["right"] = convolve(speech, rir[2])[: len(speech)]
                if np.random.rand() < self.noise_prob or not self.train:
                    out["left"] += convolve(noise, rir[1])[: len(speech)]
                    out["right"] += convolve(noise, rir[3])[: len(speech)]
                y = self.angles[int(rir_path.split("_")[-1].split(".")[0])]
            elif self.rir_filetype == "wav":
                rir_left, _ = audioread(rir_path)
                rir_right, _ = audioread(rir_path.replace("left", "right"))
                out["left"] = convolve(speech, rir_left)[: len(speech)]
                out["right"] = convolve(speech, rir_right)[: len(speech)]
                y = self.angles[rir_idx]

        norm_factor = np.maximum(
            np.max(np.abs(out["left"])), np.max(np.abs(out["right"]))
        )
        out["left"] /= norm_factor
        out["right"] /= norm_factor

        if self.save:
            print(y, rir_path, file_path, snr)
            audiowrite("sample.wav", speech, self.sample_rate)
            audiowrite(
                "sample_sim.wav",
                np.vstack((out["left"], out["right"])).T,
                self.sample_rate,
            )

        if self.binary:
            y = 1 if y > 0 else 0
        return out, y

    def __len__(self):
        return self.set_len

    def read(self, path, noise_path):
        """Read in audio samples from a list of files."""
        info = audioinfo(path)
        assert info.samplerate == self.sample_rate
        if self.fixed_duration is None:
            signal, _ = audioread(path)
            noise, _ = audioread(noise_path)
            if self.additive:
                noise = noise - signal
            return signal, noise

        duration = int(info.samplerate * self.fixed_duration)
        if info.frames < duration:  # pad
            if self.middle:
                start = int((duration - info.frames) / 2)
            else:
                start = self.random.randrange(duration - info.frames)
            signal = np.zeros(duration)
            noise = np.zeros(duration)
            tmp, _ = audioread(path)
            tmp_noise, _ = audioread(noise_path)
            signal[start : start + len(tmp)] = tmp
            noise[start : start + len(tmp)] = tmp_noise
        else:
            if self.middle:
                start = int((info.frames - duration + 1) / 2)
            else:
                start = self.random.randrange(info.frames - duration + 1)
            stop = start + duration
            signal, _ = audioread(path, start=start, stop=stop)
            noise, _ = audioread(noise_path, start=start, stop=stop)

        if self.additive:
            noise = noise - signal
        return signal, noise
