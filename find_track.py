import pandas as pd
import numpy as np
import scipy.sparse as spl
from heygoogle import HeyGoogle

g = HeyGoogle()
g.playlist_meta['pid_code'], pid_codes = g.recode(g.playlist_meta['pid'])
g.song_meta['song_code'], song_codes = g.recode(g.song_meta['song_id'])
g.tag_meta['tag_code'], tag_codes = g.recode(g.tag_meta['tag'])

## RUN the TASKS ##
all_tasks = [[100, 1000, ['S', 'ST'], 0.4, 'song'], [10, 1000, ['NT'], 0.4, 'tag']]
for i in range(1,2):
    split, knn_k, test_task, powb, origin = all_tasks[i]
    if i == 0:
        train = pd.read_csv('./all_data/train/train_playlists.csv')
        test = pd.read_csv('./all_data/val/val_playlists.csv')

        test_tasks_pids = g.val_playlist_meta[g.val_playlist_meta.task.isin(test_task)].pid.unique()
        test = test[test.pid.isin(test_tasks_pids)].copy()

        train['pid_code'] = train['pid'].map(pid_codes)
        train['song_code'] = train['song_id'].map(song_codes)
        train.sort_values(['pid_code', 'song_code'], inplace=True)

        test['pid_code'] = test['pid'].map(pid_codes)
        test['song_code'] = test['song_id'].map(song_codes)

        train_agg = train.drop_duplicates(subset=['pid_code', 'song_code']).copy()
        test_agg = test.drop_duplicates(subset=['pid_code', 'song_code']).copy()
        train_agg['val'] = 1
        test_agg['val'] = 1

        # val_stoch
        train_agg['val_stoch'] = train_agg.groupby('pid_code').val.transform(lambda x: x / np.linalg.norm(x))
        test_agg['val_stoch'] = test_agg.groupby('pid_code').val.transform(lambda x: x / np.linalg.norm(x))

        # test_agg_pop
        test_agg_pop = test_agg.join(train.song_code.value_counts().rename('pop'), on='song_code')
        test_agg_pop['pop'].fillna(1, inplace=True)

        # sp_A
        sp_A = spl.coo_matrix((train_agg['val_stoch'].values.T, train_agg[['pid_code', 'song_code']].values.T))
        sp_A._shape = (int(g.playlist_meta.pid_code.max() + 1), int(g.song_meta.song_code.max() + 1))
        sp_A = sp_A.tocsr()

        sp_A_t = sp_A.T

        sp_A_const = spl.coo_matrix((train_agg['val'].values.T, train_agg[['pid_code', 'song_code']].values.T))
        sp_A_const._shape = (int(g.playlist_meta.pid_code.max() + 1), int(g.song_meta.song_code.max() + 1))
        sp_A_const = sp_A_const.tocsr()
        sp_A_const_t = sp_A_const.T

        res = g.recs_for_ids(test_agg.pid_code.unique(), 'song', origin, test_agg_pop, powb, sp_A, sp_A_const_t, knn_k, split, pid_codes, song_codes)

    elif i == 1:
        train = pd.read_csv('./all_data/train/train_playlists_tag.csv')
        test = pd.read_csv('./all_data/val/val_playlists_tag.csv')
        train_track = pd.read_csv('./all_data/train/train_playlists.csv')

        test_tasks_pids = g.val_playlist_meta[g.val_playlist_meta.task.isin(test_task)].pid.unique()
        test = test[test.pid.isin(test_tasks_pids)].copy()

        train['pid_code'] = train['pid'].map(pid_codes)
        train['tag_code'] = train['tag'].map(tag_codes)
        train.sort_values(['pid_code', 'tag_code'], inplace=True)

        train_track['pid_code'] = train_track['pid'].map(pid_codes)
        train_track['song_code'] = train_track['song_id'].map(song_codes)
        train_track.sort_values(['pid_code', 'song_code'], inplace=True)

        test['pid_code'] = test['pid'].map(pid_codes)
        test['tag_code'] = test['tag'].map(tag_codes)

        train_agg = train.drop_duplicates(subset=['pid_code', 'tag_code']).copy()
        train_track_agg = train_track.drop_duplicates(subset=['pid_code', 'song_code']).copy()
        test_agg = test.drop_duplicates(subset=['pid_code', 'tag_code']).copy()

        train_agg['val'] = 1
        train_track_agg['val'] = 1
        test_agg['val'] = 1

        train_agg['val_stoch'] = train_agg.groupby('pid_code').val.transform(lambda x: x / np.linalg.norm(x))
        test_agg['val_stoch'] = test_agg.groupby('pid_code').val.transform(lambda x: x / np.linalg.norm(x))

        test_agg_pop = test_agg.join(train.tag_code.value_counts().rename('pop'), on='tag_code')
        test_agg_pop['pop'].fillna(1, inplace=True)

        sp_A = spl.coo_matrix((train_agg['val_stoch'].values.T, train_agg[['pid_code', 'tag_code']].values.T))

        sp_A._shape = (int(g.playlist_meta.pid_code.max() + 1), int(g.tag_meta.tag_code.max() + 1))
        sp_A = sp_A.tocsr()
        sp_A_t = sp_A.T

        sp_A_const = spl.coo_matrix(
            (train_track_agg['val'].values.T, train_track_agg[['pid_code', 'song_code']].values.T))
        sp_A_const._shape = (int(g.playlist_meta.pid_code.max() + 1), int(g.song_meta.song_code.max() + 1))
        sp_A_const = sp_A_const.tocsr()
        sp_A_const_t = sp_A_const.T

        res = g.recs_for_ids(test_agg.pid_code.unique(), 'song', origin, test_agg_pop, powb, sp_A, sp_A_const_t, knn_k, split, pid_codes, song_codes)

    g.to_json(res, 'song', i)