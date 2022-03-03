from sklearn.manifold import TSNE
from sklearn.datasets import load_digits

# MNIST 데이터 불러오기
data = load_digits()
# (1797, 64)    8x8 의 데이터가 들어가있음

print(data['target'][1700])
print(data['data'][1700])

# # 2차원으로 차원 축소
# n_components = 2

# # t-sne 모델 생성
# model = TSNE(n_components=n_components, learning_rate='auto', init='random')

# # 학습한 결과 2차원 공간 값 출력
# print(model.fit_transform(data.data))
