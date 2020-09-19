# K-NN with case by case (헤이구글)



## 모델 설명

- 플레이리스트 기반 k-Nearest Neighbor 방법

- 해당 방법은 플레이리스트와 유사한 k-Nearest-Neighbor(playlist)의 노래 혹은 태그 포함 여부를 바탕으로 노래 혹은 태그를 추천한다.

- 플레이리스트에 주어진 정보에 따라 k-Nearest Neighbor(playlist)를 구하는 방식은 아래와 같다.

  1) 곡과 태그가 모두 주어진 경우, **곡**을 기반으로 k개의 유사한 플레이리스트 정보를 이용한다.

  2) 태그만 주어진 경우에는 **태그**를 기반으로 k개의 유사한 플레이리스트 정보를 이용한다.

  3) 제목만 주어진 경우 **제목**을 기반으로 k개의 유사한 플레이리스트 정보를 이용한다.

- k-Nearest-Neighbor를 구한 뒤에는 **타겟 플레이리스트 ~ k 개 플레이리스트와의 유사도**와 **k-nearest 플레이리스트의 노래 혹은 태그 포함 여부**를 바탕으로 타겟 노래 혹은 태그의 점수를 계산한다.



## Package Requirements

- numpy ==1.19.1

- khaiii==0.4 ([https://github.com/kakao/khaiii/wiki/%EB%B9%8C%EB%93%9C-%EB%B0%8F-%EC%84%A4%EC%B9%98](https://github.com/kakao/khaiii/wiki/빌드-및-설치) 참조!)

- pandas==1.0.5

- scipy==1.5.2

  

## 모델 실행하는 방법

1. 먼저 making_csv.py 파일과 khaiiipreprocessing.py 파일을 실행시킨 이후 py 파일 실행에 필요한 csv 파일을 만듭니다．
 ( *) khaiiipreprocessing 파일은 윈도우 환경에서는 실행되지 않습니다.)

2. 예측은 크게 (1) 곡 예측 (find_track) 과 (2) 태그 예측 (find_tag)로 나뉘어져 있습니다．  

   2-1) 테스트 데이터의 예측을 위한 파일은 test_find_track.py, test_find_tag.py 입니다． 

3. heygoogle.py는 여타 py파일에서 import 하기 위한 용도로 만들었으며，HeyGoogle 클래스를 담고 있습니다.

4. 모든 예측을 마치면 result 폴더에 final_results_tag1, 2, 3 및 final_results_track1, 2, 3 총 6개의 파일이 생성됩니다． 

5. result_packing.py 파일은 test 예측파일을 합치기 위한 concat_results_final() 함수로 구성되어 있습니다. 함수를 실행시키면 됩니다． 

6. 위 과정을 마치면 result 폴더에 최종 예측 파일인 "final_results.json" 이 생성됩니다． 

   *) all_data 폴더에 대회 측에서  제공한 파일들이 들어있어야 하나,  github에 대용량의 파일을 올릴 수 없었습니다. all_data폴더에 song_meta.json,  genre_gn_all.json, val.json, test.json 파일이 들어가야 위 과정을 실행 할 수 있습니다.





## 참고사항

- Windows에서 작업을 진행했습니다. (Khaiii 모듈 다룰 떄는 Mac Os 사용)

- RAM Memory 8GB, 16GB에서 다 실행되었습니다.

- 참고 논문: 'Efficient K-NN for Playlist Continuation', Kelen et al. 2018.

  



#### copyright by 김종인, 양명한, 이경환, 홍기봉

