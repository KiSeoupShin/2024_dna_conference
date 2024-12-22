# 2024 D&A Conference
## **Text Guided ZS-CIR with Segmentation**

<br/>

## 1. 배경 & 목적
 
- 기존의 Composed Image Retrieval(CIR) 은 전체 이미지와 텍스트를 통해 검색을 수행
- 이미지 속 객체에 더욱 집중하기 위해 지역 정보를 활용하고자 함
- 이를 통해, 기존의 모델들 보다 더욱 정교한 검색이 가능할 것이라고 판단

<br/>

## 2. 프로젝트 기간

- 팀 구성 : 2024년 7월 中
- 최종 발표 : 2024년 11월 15일

<br/>

## 3. Model FLow

- Model Architecture
![image](https://github.com/user-attachments/assets/5e6f46e2-2521-49a4-88dd-b815c272d8a5)



- Training FlowChart
![image](https://github.com/user-attachments/assets/e700ed49-1405-4d5e-a4a8-cbbe67f83d47)




## 4. Model Experiments

- COCO Segmentation
  
| Method                     | R1    | R5    | R10   |
|----------------------------|-------|-------|-------|
| Pic2Word-Composed          | 11.5  | 24.8  | 33.4  |
| Pic2Word-Image             | 8.6   | 15.4  | 18.9  |
| Pic2Word-Text              | 6.1   | 15.7  | 23.5  |
| Pic2Word-Image + Text      | 10.2  | 20.2  | 26.6  |
| **Segment-Composed**       | 16.7 | **33.1** | **42.5** |
| Segment-Image              | 13.8 | 21.6 | 25.7 |
| Segment-Text               | 8.06  | 21.3 | 30.6 |
| **Segment-Image + Text**   | **17.4** | 28.5 | 34.8 |

- ImageNet-S
  
| Average                    | R1    | R5    | R10   | R50   | R100  | R200  |
|----------------------------|-------|-------|-------|-------|-------|-------|
| Pic2Word-Composed          | 1.41  | 6.71  | 10.75 | 23.36 | 30.27 | 37.86 |
| Pic2Word-Image             | 0.00  | 0.14  | 0.39  | 4.20  | 8.96  | 14.55 |
| Pic2Word-Text              | 0.14  | 0.36  | 0.49  | 2.04  | 4.19  | 8.07  |
| Pic2Word-Image + Text      | 0.00  | 0.72  | 1.63  | 11.38 | 19.75 | 28.14 |
| **Segment-Composed**       | **1.86** | **7.06** | **10.86** | **24.81** | **33.37** | **41.66** |
| Segment-Image              | 0.00  | 0.22  | 0.55  | 6.22  | 13.04 | 19.92 |
| Segment-Text               | 0.02  | 0.22  | 0.48  | 2.03  | 4.16  | 8.47  |
| **Segment-Image + Text**   | 0.10 | 3.49 | 6.49 | 22.52 | 31.44 | 38.92 |
