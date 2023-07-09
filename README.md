# 🔍 프로젝트 개요
- **프로젝트 목표**

    우리 몸의 구조와 기능에 중요한 영향을 미치는 뼈 분할 문제를 수행한다. 딥러닝 기술의 Semantic Segmentation 모델을 활용하여 손가락, 손등, 팔을 포함한 손 부분의 뼈를 분할한다. 질병 진단, 수술 계획, 의료 장비 제작, 의료 교육에 활용될 수 있을 것이라 예상한다.
    

- **Input :** hand bone x-ray 이미지(사람 별로 두 장의 이미지가 존재), segmentation annotation json file
- **Output :** 모델은 각 픽셀 좌표에 따른 class를 출력하며, 이를 rle로 변환하여 제출


# 🔥 수행 내용 🔥
1. EDA, Dataset Split, 데이터 셋 전수 조사
2. 성능 향상 방법 구현
    - Classification Head 연결 
    - 손등 Classifier 개별 구성 (29 개 중 8개 Class)
3. Loss 실험
4. Augmentation 실험
5. 모델 예측 결과 분석 후 실험
    - 모델이 잘 예측하지 못하는 부분을 보완해주기 위한 실험
6. 성능 향상을 위한 접근 방법 
    - ensemble (soft voting, hard voting) 수행
    - pseudo labeling
    - TTA (Test Time Augmentation)
# 🧱 협업 문화
- 협업 tool : Github, WandB, Notion, kakaoTalk, zoom, slack
    - Github : Semantic Segmentation 프로젝트 코드 관리
    - WandB : 모델 실험 결과 공유, Train Loss 비교
    - Notion :  진행사항 기록, 모델 훈련 중 발생한 error 기록
- Githup branch
    - 적용하고자 하는 기능마다 branch를 생성하여 실험을 진행
        
# 🗝️최종 모델
- **model**
    - Unet ++ / maxvit-B
- **Augmentation**
    - HorizontalFlip (class 29)
    - geometric.rotate.Rotate (class 29)
    - CropNonEmptyMaskIfExists (class 29)
- **Optimizer, Scheduler**
    - AdamW
    - CosineAnnealingLR
- **각 fold의 손등 class(성능이 낮은 class) 모델을 soft voting ensemble로 성능 향상 후 5개의 fold를 hard voting ensemble로 최종 성능 향상**
    ![Alt text](image.png)
# 🏆Result
- public(0.9726) 9등
- private(0.9733) 최종 8등

![Alt text](image-1.png)

![Alt text](image-2.png)


# 👨‍🌾  팀 구성 및 역할
- 도환 : Data EDA 및 전수 조사, TTA, soft voting ensemble
- 아라 : smp library를 이용한 실험, augmentation 실험, loss 실험
- 성운 : 손등 classifier training, classfication head, soft voting ensemble, pseudo labeling
- 현민 : 손등 classifier inference, Sliding window, Hard voting ensemble, augmentation 실험


|김도환 |                                                  서아라|                                                   조성운|한현민|
|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| [<img src="https://avatars.githubusercontent.com/u/121927513?v=4" alt="" style="width:100px;100px;">](https://github.com/rlaehghks5) <br/> | [<img src="https://avatars.githubusercontent.com/u/68554446?v=4" alt="" style="width:100px;100px;">](https://github.com/araseo) <br/>  |[<img src="https://avatars.githubusercontent.com/u/126544082?v=4" alt="" style="width:100px;100px;">](https://github.com/nebulajo) <br/> | [<img src="https://avatars.githubusercontent.com/u/33598545?s=400&u=d0aaa9e96fd2fa1d0c1aa034d8e9e2c8daf96473&v=4" alt="" style="width:100px;100px;">](https://github.com/Hyunmin-H) <br/> |

****