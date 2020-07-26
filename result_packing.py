import pandas as pd
from heygoogle import HeyGoogle



def concat_results():
    res1 = pd.read_json('./result/results_track_1.json')
    res2 = pd.read_json('./result/results_track_2.json')
    res3 = pd.read_json('./result/results_track_3.json')
    res4 = pd.read_json('./result/results_tag_1.json')
    res5 = pd.read_json('./result/results_tag_2.json')
    res6 = pd.read_json('./result/results_tag_3.json')

    res_track = pd.concat([res1, res2, res3], axis=0)
    res_tag = pd.concat([res4, res5, res6], axis=0)
    res_track.drop(columns='tags', inplace=True)
    res_tag.drop(columns='songs', inplace=True)

    result = pd.merge(res_track, res_tag, on='id')
    missing_list = result['id']
    missing_df = g.val.loc[~g.val['id'].isin(missing_list)]

    missing_df['tags'] = [g.pop_tags for _ in range(len(missing_df))]
    missing_df['songs'] = [g.pop_songs for _ in range(len(missing_df))]
    missing_df.drop(columns=['plylst_title', 'like_cnt', 'updt_date'], inplace=True)

    result = pd.concat([result, missing_df], axis=0)
    result.to_json('./result/results.json', orient='records')

def concat_results_final():
    res1 = pd.read_json('./result/final_results_track_1.json')
    res2 = pd.read_json('./result/final_results_track_2.json')
    res3 = pd.read_json('./result/final_results_track_3.json')
    res4 = pd.read_json('./result/final_results_tag_1.json')
    res5 = pd.read_json('./result/final_results_tag_2.json')
    res6 = pd.read_json('./result/final_results_tag_3.json')

    res_track = pd.concat([res1, res2, res3], axis=0)
    res_tag = pd.concat([res4, res5, res6], axis=0)
    res_track.drop(columns='tags', inplace=True)
    res_tag.drop(columns='songs', inplace=True)

    result = pd.merge(res_track, res_tag, on='id')
    missing_list = result['id']
    missing_df = g.test.loc[~g.test['id'].isin(missing_list)]

    missing_df['tags'] = [g.pop_tags for _ in range(len(missing_df))]
    missing_df['songs'] = [g.pop_songs for _ in range(len(missing_df))]
    missing_df.drop(columns=['plylst_title', 'like_cnt', 'updt_date'], inplace=True)

    result = pd.concat([result, missing_df], axis=0)
    result.to_json('./result/final_results.json', orient='records')

if __name__ == '__main__':
    g = HeyGoogle()
    # concat_results()
    concat_results_final()