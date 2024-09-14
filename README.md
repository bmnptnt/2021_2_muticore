# 2021_2_multicore
## OpenCL 기반의 병렬화를 통한 CNN 가속화 프로젝트
- Environment : C, Visual Studio, OpenCL
- Model : VGG16, Dataset : CIFAR-10
- convolution layer의 기본적인 병렬화와 BATCH를 통해 가속화 하였습니다. 데이터 흐름 개선, 타일링 기법과 같은 것은 도입하지 못하였기에 모범적인 성능 향상은 아닙니다.
##### *__※세종대학교 멀티코어프로그래밍 수강하는 학생들은 코드를 복사해서 그대로 제출하는 일이 없길 바랍니다. 0점 처리 되거니와 높은 성능이 아니라 코드 복사가 걸리지 않더라도 이상적인 점수를 받기 어려울 것입니다.__*
