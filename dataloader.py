from torch.utils.data import Dataset, DataLoader
import pandas as pd
import librosa
import os
import torch


class MelodyDataset(Dataset):
    def __init__(self, metadata_path='metadata.csv', audio_dir='audio', sr=22050, split='training', num_label=40):
        super(MelodyDataset, self).__init__()
        
        self.metadata_path = metadata_path
        self.audio_dir = audio_dir
        self.sr = sr
        self.split = split
        self.num_label = num_label
        self.mapping_dict = self.__mapping_dict__()
        self.track_column_name = 'track_id'
        
        self.metadata_df = pd.read_csv(metadata_path)
        # return only split
        self.metadata_df = self.metadata_df[self.metadata_df['split'] == self.split]

    @staticmethod
    def __mapping_dict__():
        melody_dict = {}
        for i in range(39):
            melody_dict[i] = i % 12
        melody_dict[39] = 12
        return melody_dict

    def _abstract_label(self, label_list):
        if self.num_label == 13:
            label = torch.Tensor([self.mapping_dict[l] for l in label_list]).long()
        elif self.num_label == 40:
            label = torch.Tensor(label_list).long()
        else:
            print("!! define num_label in the right manner !!")
            raise NotImplementedError
        return label

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        track_id = self.metadata_df.iloc[idx][self.track_column_name]
        label = self._abstract_label(self.metadata_df.iloc[idx].values[2:].tolist())

        # to compute l_voice, is_voice separates total label to voice/non-voice
        is_voice = label.clone()
        # change the original label to voice/non-voice label (0-38 -> 1, 39 -> 0)
        for i in range(is_voice.shape[0]):
            if is_voice[i] == (self.num_label-1):
                is_voice[i] = 0
            else:
                is_voice[i] = 1

        audio, sr = librosa.load(os.path.join(self.audio_dir, "%05d.wav" % track_id), sr=self.sr)
                                 
        return audio, label, is_voice


if __name__ == '__main__':

    dataset = MelodyDataset(metadata_path='metadata.csv', audio_dir='audio', sr=22050, split='training', num_label=40)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    first_batch = train_loader.__iter__().__next__()
    audio, label, is_voice = first_batch
    print(audio.shape)
    print(label.shape)

