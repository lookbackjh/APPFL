import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# 모델 정의 (변경 없음)
class QuadraticModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QuadraticModel, self).__init__()
        self.layer1 = nn.Linear(input_dim, 32)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(32, 16)
        self.output_layer = nn.Linear(16, output_dim)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.output_layer(x)
        return x

# ADMM 트레이너 클래스 (변경 없음)
class ADMMTrainer:
    def __init__(self, model, global_model, dual_vars, device, rho=1.0, num_epochs=100, learning_rate=0.01):
        self.model = model
        self.global_model = global_model
        self.dual_vars = dual_vars
        self.rho = rho
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.device = device

    def train(self, train_loader):
        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        global_params = {name: param.detach() for name, param in self.global_model.named_parameters()}
        
        for epoch in range(self.num_epochs):
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                admm_penalty = 0.0
                for name, param in self.model.named_parameters():
                    if name in global_params:
                        # 사용자가 추가했던 admm_penalty += torch.sum(...) 부분은 표준 ADMM 공식이 아니므로 제거했습니다.
                        admm_penalty += (self.rho / 2) * torch.norm(param - global_params[name] + self.dual_vars[name])**2
                total_loss = loss + admm_penalty
                total_loss.backward()
                optimizer.step()
        
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                grad=param.grad
                param_state= optimizer.state[param]
                if name in global_params:
                    self.dual_vars[name].add_(param - global_params[name])
        
        return self.model.state_dict()

# 데이터셋 클래스 (변경 없음)
class SimpleDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)

# --- IID 데이터 분배를 위한 수정 (핵심) ---
# 함수의 이름 변경 및 정렬(sort) 로직 제거
def create_iid_datasets(num_clients, num_samples, batch_size):
    # 실제 2차 함수 관계 정의
    def true_f(x):
        return 0.5 * x**2 + 2.0 * x - 3.0

    # -10부터 10까지 넓은 범위의 X 데이터 생성 (이미 무작위)
    X = np.random.uniform(-10, 10, (num_samples, 1))
    y = true_f(X) + np.random.randn(num_samples, 1) * 2.5
    
    # --- 핵심 변경: 데이터를 정렬하는 부분을 제거 ---
    # 이제 데이터는 완전히 무작위로 섞인 상태입니다.
    
    # 클라이언트별 DataLoader 생성
    client_dataloaders = []
    samples_per_client = num_samples // num_clients
    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = (i + 1) * samples_per_client
        # 정렬되지 않은 원본 데이터에서 그대로 잘라냄
        client_X, client_y = X[start_idx:end_idx], y[start_idx:end_idx]
        dataset = SimpleDataset(client_X, client_y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        client_dataloaders.append(loader)
        
    return client_dataloaders, (X, y), true_f

# --- 학습 준비 ---
input_dim = 1
output_dim = 1
client_num = 5

# IID 데이터셋 생성 함수 호출
train_loaders, full_data, true_function = create_iid_datasets(client_num, num_samples=1000, batch_size=256)

# 모델 생성 (변경 없음)
global_model = QuadraticModel(input_dim, output_dim)
client_models = [QuadraticModel(input_dim, output_dim) for _ in range(client_num)]
dual_vars = [{name: torch.zeros_like(param) for name, param in client_model.named_parameters()} for client_model in client_models]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

global_model.to(device)
for client_model in client_models:
    client_model.to(device)
for dual_var in dual_vars:
    for key in dual_var:
        dual_var[key] = dual_var[key].to(device)

# --- 메인 학습 루프 (변경 없음) ---
# IID 환경에서는 문제가 훨씬 쉬워졌으므로 라운드 수를 줄여도 결과가 잘 나옵니다.
for j in range(100):
    print(f"--- Round {j+1}/100 ---")
    for i in range(client_num):
        admm_trainer = ADMMTrainer(client_models[i], global_model, dual_vars[i], device, rho=0.01, num_epochs=50, learning_rate=0.001)
        admm_trainer.train(train_loaders[i])
        
    global_dict = global_model.state_dict()
    for key in global_dict.keys():
        global_dict[key] = torch.mean(torch.stack([client_models[i].state_dict()[key] for i in range(client_num)]), dim=0)
    global_model.load_state_dict(global_dict)

    # print test loss after each round
    global_model.eval()
    criterion = nn.MSELoss()
    with torch.no_grad():
        X_full = torch.tensor(full_data[0], dtype=torch.float32).to(device)
        y_full = torch.tensor(full_data[1], dtype=torch.float32).to(device)
        y_pred = global_model(X_full)
        loss = criterion(y_pred, y_full)
        print(f"Test Loss after round {j+1}: {loss.item()}")

# --- 결과 확인 및 시각화 (변경 없음) ---
print("\n--- Visualization of Learning Results ---")
plt.figure(figsize=(10, 6))

X_all, y_all = full_data
plt.scatter(X_all, y_all, alpha=0.2, label='All Client Data')

x_line = torch.linspace(-10, 10, 200).reshape(-1, 1)
y_true_line = true_function(x_line.numpy())
plt.plot(x_line.numpy(), y_true_line, color='red', linestyle='--', linewidth=3, label='True Function: y=0.5x^2+2x-3')

global_model.to('cpu')
with torch.no_grad():
    y_pred_line = global_model(x_line)
plt.plot(x_line.numpy(), y_pred_line.numpy(), color='green', linewidth=3, label='Learned Global Model')

plt.title('Federated Learning on IID Quadratic Data')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('federated_learning_quadratic.png')