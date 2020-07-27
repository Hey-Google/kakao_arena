import pandas as pd
import numpy as np

class HeyGoogle:
    def __init__(self):
        self._dir = './all_data/melon_give_us/'
        self.song_meta = pd.read_json(self._dir + 'song_meta.json', encoding='utf-8')
        self.genre_gn_all = pd.read_json(self._dir + 'genre_gn_all.json', encoding='utf-8', typ='series')
        self.train = pd.read_json(self._dir + 'train.json', encoding='utf-8')
        self.train2 = pd.read_json(self._dir + 'train.json', encoding='utf-8')
        self.val = pd.read_json(self._dir + 'val.json', encoding='utf-8')
        self.test = pd.read_json(self._dir + 'test.json', encoding='utf-8')

        ### for name_track ###
        self.train_names = pd.read_csv('./all_data/train/train_names.csv')
        self.train_names = self.train_names.dropna()
        self.val_names = pd.read_csv('./all_data/val/val_names.csv')
        self.val_names = self.val_names.dropna()
        self.test_names = pd.read_csv('./all_data/test/test_names.csv')
        self.test_names = self.test_names.dropna()

        self.train_names.name_normalized = self.train_names.name_normalized.astype('str')
        self.train_names.name_normalized = self.train_names.name_normalized.apply(lambda x: x.split())
        self.val_names.name_normalized = self.val_names.name_normalized.astype('str')
        self.val_names.name_normalized = self.val_names.name_normalized.apply(lambda x: x.split())
        self.test_names.name_normalized = self.test_names.name_normalized.astype('str')
        self.test_names.name_normalized = self.test_names.name_normalized.apply(lambda x: x.split())

        self.train_name = [name for l in self.train_names.name_normalized for name in l]
        self.val_name = [name for l in self.val_names.name_normalized for name in l]
        self.tot_name = self.train_name + self.val_name

        self.test_name = [name for l in self.test_names.name_normalized for name in l]
        self.test_tot_name = self.train_name + self.test_name

        self.train_name_set = set([token for l in self.train_names.name_normalized for token in l])
        self.val_name_set = set([token for l in self.val_names.name_normalized for token in l])
        self.test_name_set = set([token for l in self.test_names.name_normalized for token in l])

        self.name_union = self.train_name_set.union(self.val_name_set)
        self.name_union = self.name_union.difference({''})
        self.name_union_list = sorted(list(self.name_union))

        self.test_name_union = self.train_name_set.union(self.test_name_set)
        self.test_name_union = self.test_name_union.difference({''})
        self.test_name_union_list = sorted(list(self.test_name_union))

        ### preprocess some files ###
        self.song_meta = pd.DataFrame({
            'song_id': self.song_meta.id,
            'album_name': self.song_meta.album_name,
            'song_name': self.song_meta.song_name
        })

        self.train_playlist_meta = pd.DataFrame({
            'pid': self.train.id,
            'name': self.train.plylst_title,
            'num_tracks': self.train.songs.apply(len),
            'num_tags': self.train.tags.apply(len)
        })

        self.val_playlist_meta = pd.DataFrame({
            'pid': self.val.id,
            'name': self.val.plylst_title,
            'num_samples': self.val.songs.apply(len),
            'num_tags': self.val.tags.apply(len)
        })

        self.test_playlist_meta = pd.DataFrame({
            'pid': self.test.id,
            'name': self.test.plylst_title,
            'num_samples': self.test.songs.apply(len),
            'num_tags': self.test.tags.apply(len)
        })

        self.name_meta = pd.DataFrame({
            'name': self.name_union_list
        })

        self.test_name_meta = pd.DataFrame({
            'name': self.test_name_union_list
        })

        ## popular songs , popular tags for padding ##
        self.pop_songs = pd.Series([s for l in self.train2.songs for s in l]).value_counts().head(100).index.tolist()
        self.pop_tags = pd.Series([s for l in self.train2.tags for s in l]).value_counts().head(10).index.tolist()

        ## _meta definition ##
        self.playlist_meta = pd.concat([self.train_playlist_meta, self.val_playlist_meta], axis=0, ignore_index=True)
        self.train_tag_set = set([tag for l in self.train.tags for tag in l])
        self.val_tag_set = set([tag for l in self.val.tags for tag in l])
        self.tag_union = self.train_tag_set.union(self.val_tag_set)
        self.tag_union = self.tag_union.difference({''})
        self.tag_union_list = sorted(list(self.tag_union))
        self.tag_meta = pd.DataFrame({'tag': self.tag_union_list})

        self.playlist_meta_final = pd.concat([self.train_playlist_meta, self.test_playlist_meta], axis=0, ignore_index=True)
        self.test_tag_set = set([tag for l in self.test.tags for tag in l])
        self.test_tag_union = self.train_tag_set.union(self.test_tag_set)
        self.test_tag_union = self.test_tag_union.difference({''})
        self.test_tag_union_list = sorted(list(self.test_tag_union))
        self.test_tag_meta = pd.DataFrame({'tag': self.test_tag_union_list})

        ## TASK defined ##
        self.val_playlist_meta['task'] = ''
        self.val_playlist_meta.loc[(self.val_playlist_meta.num_samples > 0) & (self.val_playlist_meta.num_tags == 0), 'task'] = 'S'
        self.val_playlist_meta.loc[(self.val_playlist_meta.num_samples > 0) & (self.val_playlist_meta.num_tags > 0), 'task'] = 'ST'
        self.val_playlist_meta.loc[(self.val_playlist_meta.name != '') & (self.val_playlist_meta.num_tags == 0), 'task'] = 'N'
        self.val_playlist_meta.loc[(self.val_playlist_meta.name != '') & (self.val_playlist_meta.num_tags > 0), 'task'] = 'NT'

        ## test TASK defined ##
        self.test_playlist_meta['task'] = ''
        self.test_playlist_meta.loc[(self.test_playlist_meta.num_samples > 0) & (self.test_playlist_meta.num_tags == 0), 'task'] = 'S'
        self.test_playlist_meta.loc[(self.test_playlist_meta.num_samples > 0) & (self.test_playlist_meta.num_tags > 0), 'task'] = 'ST'
        self.test_playlist_meta.loc[(self.test_playlist_meta.name != '') & (self.test_playlist_meta.num_tags == 0), 'task'] = 'N'
        self.test_playlist_meta.loc[(self.test_playlist_meta.name != '') & (self.test_playlist_meta.num_tags > 0), 'task'] = 'NT'

    ## functions defined ##
    def recode(self, column, min_val=0):
        uniques = column.unique()
        codes = range(min_val, len(uniques) + min_val)
        code_map = dict(zip(uniques, codes))
        return (column.map(code_map), code_map)

    def reverse_code(self, column, code_map):
        inv_map = {v: k for k, v in code_map.items()}
        return column.map(inv_map)

    def recs_for_ids(self, ids_, predict_type, origin, test_agg_pop, powb, sp_A, sp_A_const_t, knn_k, split, pid_codes, what_codes, plusadd=0, is_test=False):
        dfs = []
        ndcgs = []
        if predict_type == 'song':
            for i, pid_ in enumerate(ids_):
                if i % 1000 == 0:
                    print(i)
                p1 = test_agg_pop[(test_agg_pop.pid_code == pid_)]  # 데이터 프레임

                if is_test == True:
                    if origin == 'song':
                        np_p1 = np.zeros([int(self.song_meta.song_code.max() + 1), 1])  # 전체 song 개수 만큼의 영벡터
                        np_p1[p1.song_code.values] = p1[['val_stoch']].values / ((p1[['pop']].values - 1) ** (powb) + 1)  # 곡 id 인덱스에다가 값을 넣는다.
                    elif origin == 'tag':
                        np_p1 = np.zeros([int(self.test_tag_meta.tag_code.max() + 1), 1])  # 전체 song 개수 만큼의 영벡터
                        np_p1[p1.tag_code.values] = p1[['val_stoch']].values / ((p1[['pop']].values - 1) ** (powb) + 1)
                    else:
                        np_p1 = np.zeros([int(self.test_name_meta.name_code.max() + 1), 1])  # 전체 name 개수 만큼의 영벡터
                        np_p1[p1.name_code.values] = p1[['val_stoch']].values / ((p1[['pop']].values - 1) ** (powb) + 1)
                else:
                    if origin == 'song':
                        np_p1 = np.zeros([int(self.song_meta.song_code.max() + 1), 1])  # 전체 song 개수 만큼의 영벡터
                        np_p1[p1.song_code.values] = p1[['val_stoch']].values / ((p1[['pop']].values - 1) ** (powb) + 1)  # 곡 id 인덱스에다가 값을 넣는다.
                    elif origin == 'tag':
                        np_p1 = np.zeros([int(self.tag_meta.tag_code.max() + 1), 1])  # 전체 song 개수 만큼의 영벡터
                        np_p1[p1.tag_code.values] = p1[['val_stoch']].values / ((p1[['pop']].values - 1) ** (powb) + 1)
                    else:
                        np_p1 = np.zeros([int(self.name_meta.name_code.max() + 1), 1])  # 전체 name 개수 만큼의 영벡터
                        np_p1[p1.name_code.values] = p1[['val_stoch']].values / ((p1[['pop']].values - 1) ** (powb) + 1)

                ### simpls : train의 모든 플레이리스트와 IIF 유사도 계산 -> Inverse Item Frequency 고려한 유사도 나옴 ###
                # 차원 : 2
                # shape : (train의 playlist 길이, 1)
                simpls = sp_A.dot(np_p1)

                ### simpls2 : 수정된 유사도 - normalization - amplification ###
                simpls2 = np.zeros_like(simpls)  # 같은 모양의 영벡터
                inds = simpls.reshape(-1).argsort()[-knn_k:][
                       ::-1]  # 행벡터로 바꾸고(reshape) - 오름차순으로 인덱스 반환(argsort) - 가장 큰 k개의 인덱스만 남기고 - 순서 바꾸기
                vals = simpls[inds]  # 상위 knn_k개 유사도를 반환
                m = np.max(vals)
                if (m == 0):
                    m += 0.01
                vals2 = ((vals - np.min(vals)) * (1 / m) + plusadd) ** 2  # normalization - amplification 후 상위 knn_개 유사도
                simpls2[inds] = vals2  # 수정된 유사도

                ### tmp : 전체 곡별 점수 ###
                tmp = sp_A_const_t[:, inds].dot(vals2)

                ### indices_np : 가장 유사한 곡/태그 100/10개의 index ###
                if origin == 'song':
                    indices_np = tmp.reshape(-1).argsort()[-(100 + split):][::-1]
                    indices_np = indices_np[np.isin(indices_np, p1.song_code) == False][:100]  # 기존 곡에 있는 것은 빼고 100개
                else:
                    indices_np = tmp.reshape(-1).argsort()[-(100):][::-1]

                dfs.append(pd.DataFrame({
                    'pid': np.repeat(pid_, 100),
                    'pos': np.arange(100),
                    'song_id': indices_np,
                    'score': tmp[indices_np, 0]
                }))

            recdf = pd.concat(dfs, axis=0)
            recdf['pid'] = self.reverse_code(recdf['pid'], pid_codes)
            recdf['song_id'] = self.reverse_code(recdf['song_id'], what_codes)
            return (recdf, ndcgs)

        elif predict_type == 'tag':
            for i, pid_ in enumerate(ids_):
                if i % 1000 == 0:
                    print(i)

                p1 = test_agg_pop[(test_agg_pop.pid_code == pid_)]

                if is_test == True:
                    if origin == 'song':
                        np_p1 = np.zeros([int(self.song_meta.song_code.max() + 1), 1])  # 전체 song 개수 만큼의 영벡터
                        np_p1[p1.song_code.values] = p1[['val_stoch']].values / ((p1[['pop']].values - 1) ** (powb) + 1)  # 곡 id 인덱스에다가 값을 넣는다.
                    elif origin == 'tag':
                        np_p1 = np.zeros([int(self.test_tag_meta.tag_code.max() + 1), 1])  # 전체 song 개수 만큼의 영벡터
                        np_p1[p1.tag_code.values] = p1[['val_stoch']].values / ((p1[['pop']].values - 1) ** (powb) + 1)
                    else:
                        np_p1 = np.zeros([int(self.test_name_meta.name_code.max() + 1), 1])  # 전체 name 개수 만큼의 영벡터
                        np_p1[p1.name_code.values] = p1[['val_stoch']].values / ((p1[['pop']].values - 1) ** (powb) + 1)
                else:
                    if origin == 'song':
                        np_p1 = np.zeros([int(self.song_meta.song_code.max() + 1), 1])
                        np_p1[p1.song_code.values] = p1[['val_stoch']].values / ((p1[['pop']].values - 1) ** (powb) + 1)
                    elif origin == 'tag':
                        np_p1 = np.zeros([int(self.tag_meta.tag_code.max() + 1), 1])
                        np_p1[p1.tag_code.values] = p1[['val_stoch']].values / ((p1[['pop']].values - 1) ** (powb) + 1)
                    else:
                        np_p1 = np.zeros([int(self.name_meta.name_code.max() + 1), 1])  # 전체 name 개수 만큼의 영벡터
                        np_p1[p1.name_code.values] = p1[['val_stoch']].values / ((p1[['pop']].values - 1) ** (powb) + 1)

                simpls = sp_A.dot(np_p1)
                simpls2 = np.zeros_like(simpls)
                inds = simpls.reshape(-1).argsort()[-knn_k:][::-1]
                vals = simpls[inds]
                m = np.max(vals)
                if (m == 0):
                    m += 0.01
                vals2 = ((vals - np.min(vals)) * (1 / m) + plusadd) ** 2
                simpls2[inds] = vals2

                ### tmp : 전체 태그별 점수 ###
                tmp = sp_A_const_t[:, inds].dot(vals2)

                ### indices_np : 가장 유사한 태그 10개의 index ###
                if origin == 'song':
                    indices_np = tmp.reshape(-1).argsort()[-(10):][::-1]
                elif origin == 'tag':
                    indices_np = tmp.reshape(-1).argsort()[-(10 + split):][::-1]
                    indices_np = indices_np[np.isin(indices_np, p1.tag_code) == False][:10]
                else:
                    indices_np = tmp.reshape(-1).argsort()[-(10):][::-1]

                dfs.append(pd.DataFrame({
                    'pid': np.repeat(pid_, 10),
                    'pos': np.arange(10),
                    'tag_id': indices_np,
                    'score': tmp[indices_np, 0]
                }))

            recdf = pd.concat(dfs, axis=0)
            recdf['pid'] = self.reverse_code(recdf['pid'], pid_codes)
            recdf['tag'] = self.reverse_code(recdf['tag_id'], what_codes)
            return (recdf, ndcgs)

    def to_json(self, res, type, num):
        if type == 'song':
            id_list = res[0].groupby('pid')['song_id'].apply(lambda x : list(x)).reset_index()
            id_list.columns = ['id', 'songs']
            id_list['tags'] = None
            id_list.to_json(f'./result/results_track_{num + 1}.json', orient='records')
        elif type == 'tag':
            id_list = res[0].groupby('pid')['tag'].apply(lambda x: list(x)).reset_index()
            id_list.columns = ['id', 'tags']
            id_list['songs'] = None
            id_list.to_json(f'./result/results_tag_{num + 1}.json', orient='records')

    def to_json_final(self, res, type, num):
        if type == 'song':
            id_list = res[0].groupby('pid')['song_id'].apply(lambda x : list(x)).reset_index()
            id_list.columns = ['id', 'songs']
            id_list['tags'] = None
            id_list.to_json(f'./result/final_results_track_{num + 1}.json', orient='records')
        elif type == 'tag':
            id_list = res[0].groupby('pid')['tag'].apply(lambda x: list(x)).reset_index()
            id_list.columns = ['id', 'tags']
            id_list['songs'] = None
            id_list.to_json(f'./result/final_results_tag_{num + 1}.json', orient='records')