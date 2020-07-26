# K-NN with case by case (헤이구글)



## 모델 설명

KNN을 활용하여 여러 케이스를  



## Package Requirements

- numpy ==1.19.1
- khaiii==0.4 ([https://github.com/kakao/khaiii/wiki/%EB%B9%8C%EB%93%9C-%EB%B0%8F-%EC%84%A4%EC%B9%98](https://github.com/kakao/khaiii/wiki/빌드-및-설치) 참조!)
- pandas==1.0.5
- scipy==1.5.2

## 모델 실행하는 방법

1. 먼저 making._csv.py 파일을 실행시킨 이후 py 파일 실행에 필요한 csv 파일을 만듭니다．(khaiiipreprocessing.py 파일은 태그 및 플레이리스트 제목을 전처리 하기위한 파일입니다.) 

2. 예측은 크게 (1) 곡 예측 (find_track) 과 (2) 태그 예측 (find_tag)로 나뉘어져 있습니다． 

   2-1) Validation 읕 위한 파일은 find_track.py, find_tag. py 입니다． 

   2-2) Test를 위한 파일은 test_find_track.py, test_find_tag.py 입니다． 

3. heygoogle.py는 여타 py파일에서 import 하기 위한 용도로 만들었으며，HeyGoogle 클래스를 담고 있습니다.

4. 모든 예측을 마치면 resutt 폴더에 Validation 6개， Test 6개 총 12개의 파일이 생성됩니다． 

   4-1) result_packing.py 파일은 validation 예측파일을 합치기 위한 concat_results() 함수와 test 예측파일을 합치기 위한 concat_results_final() 함수로 구성되어 있습니다．용도에 알맞은 함수를 실행시키면 됩니다． 

5. 위 과정을 마치연 result 폴더에 최종 파일이 생성됩니다． Validation은 results.json, test는 final_results.json이 생성됩니다． 

   *) all_data 폴더에 대회 측에서  제공한 파일들이 들어있어야 하나,  github에 대용량의 파일을 올릴 수 없었습니다. all_data폴더에 song_meta.json,  genre_gn_all.json, val.json, test.json 파일이 들어가야 위 과정을 실행 할 수 있습니다.





#### copyright by 김종인, 양명한, 이경환, 홍기봉

