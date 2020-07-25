import pandas as pd
import json
import csv

def load_data():
    train = open('./all_data/melon_give_us/train.json', encoding='utf-8')
    train = json.load(train)
    song_meta = open('./all_data/melon_give_us/song_meta.json', encoding='utf-8')
    song_meta = json.load(song_meta)
    val = open('./all_data/melon_give_us/val.json', encoding='utf-8')
    val = json.load(val)
    return train, song_meta, val

def make_train_plyst_csv(file):
    plyst_song = [['pid', 'song_id']]
    plyst_tag = [['pid', 'tag']]
    for f in file:
        for song in f['songs']:
            plyst_song.append([f['id'], song])
        for tag in f['tags']:
            plyst_tag.append([f['id'], tag])
    with open('./all_data/train/train_playlists.csv', 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(plyst_song)
    with open('./all_data/train/train_playlists_tag.csv', 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(plyst_tag)

def make_val_plyst_csv(file):
    plyst_song = [['pid', 'song_id']]
    plyst_tag = [['pid', 'tag']]
    for f in file:
        for song in f['songs']:
            plyst_song.append([f['id'], song])
        for tag in f['tags']:
            plyst_tag.append([f['id'], tag])
    with open('./all_data/val/val_playlists.csv', 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(plyst_song)
    with open('./all_data/val/val_playlists_tag.csv', 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(plyst_tag)


if __name__ == '__main__':
    train, song_meta, val = load_data()
    make_train_plyst_csv(train)
    make_val_plyst_csv(val)