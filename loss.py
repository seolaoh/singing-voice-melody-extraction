import torch
from torch import nn
import math

class CELossWithGSL(nn.Module):
    def __init__(self, num_label, blur_range=3): # 논문에서 3으로 했음 블러레인지 변경 가능 (2~4,5?)
        super(CELossWithGSL, self).__init__()
        self.dim = -1
        self.num_label = num_label
        self.blur_range = blur_range
        self.gaussian_decays = [gaussian_val(dist=d) for d in range(blur_range + 1)]  # 리스트

    @staticmethod    # 논문의 함수 : 각 위치에 따른 값 계산 , standard normal 가정 시그마 : 1 시그마도 변경 가능할듯?
    def gaussian_val(dist: int, sigma = 1):
        return math.exp(-math.pow(dist, 2) / (2 * math.pow(sigma, 2)))

    # Gaussian smoothed label과 log softmax(예측값)의 CE를 구함
    def forward(self, pred: torch.Tensor, label: torch.Tensor):
        # pred: (b, 240, num_label)  # target: (b, 240)
        pred_logit = torch.log_softmax(pred, dim=self.dim)  # out: (b, 240, num_label)
        label_smoothed = smoothed_label(label, self.num_label)

        # 240 프레임 모두에 대해 CE 계산 합 -> 평균
        target_loss_sum = -(pred_logit * label_smoothed).sum(dim=self.dim)
        return target_loss_sum.mean()


    # 가우시안 스무딩 처리된 라벨 리턴 out: (b, 240, num_label)
    def smoothed_label(self, label: torch.Tensor, num_label):
        label_onehot = torch.FloatTensor(*(label.size() + (num_label,))).zero_().to(label.device)
        with torch.no_grad():
            # 라벨 인덱스가 4보다 작거나 36보다 클 때 오버라이드 방지하기 위해서 3 -> 0 (역방향)으로 for 문 돌린다.
            #  2일 때, 2 1 0 0 vs. 0 0 1 2 차이 뭔지 잘 모르겠네..
            for dist in range(self.blur_range, -1, -1):
                for direction in [1, -1]:
                    # 블러 인덱스 :  타겟 인덱스에서 + - 3 한 것 까지 (3이니까 총 7개)
                    # clamp : 최대 최소 설정 범위로 값을 찝어준다. 최소 0 최대 39
                    blur_idx = torch.clamp(label + (direction * dist), min=0, max=self.num_label - 1)
                    decayed_val = self.gaussian_decays[dist]
                    # 원핫 벡터의 블러 인덱스 자리에 decayed value 넣어준다.
                    label_smoothed = label_onehot.scatter_(dim=1, index=torch.unsqueeze(blur_idx, dim=2), value=decayed_val) # 디멘션 문제
        return label_smoothed


# 여기는 예시
def test_gaussian_blur():
    loss = CELossWithGSL(num_label=40)
    print(loss.smoothed_label(torch.LongTensor(2, 1) % 40))
if __name__ == '__main__':
    test_gaussian_blur()