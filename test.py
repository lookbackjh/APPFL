from mpi4py import MPI
import torch
import numpy as np # mpi4py가 텐서를 보낼 때 내부적으로 필요할 수 있음

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# --- 서버의 역할 ---
if rank == 0:
    requests = []
    received_tensors = [None] * size

    print(f"[Rank {rank} / 서버]: 모든 클라이언트로부터 GPU 텐서 수신을 시작합니다...")
    for i in range(1, size):
        # 수신할 텐서를 담을 빈 GPU 텐서를 준비할 수도 있지만, 
        # mpi4py 3.0 이상에서는 자동으로 메모리를 할당해줍니다.
        req = comm.irecv(source=i, tag=11)
        requests.append(req)

    for i in range(1, size):
        client_rank = i
        received_tensor = requests[i-1].wait()
        print(f"[Rank {rank} / 서버]: Rank {client_rank}로부터 데이터를 받았습니다.")
        print(f"    - 받은 텐서의 일부: {received_tensor[:5]}")
        print(f"    - 받은 텐서의 위치(device): {received_tensor.device}")
        received_tensors[i] = received_tensor
        
    print("\n[Rank {rank} / 서버]: 모든 클라이언트로부터 데이터를 성공적으로 받았습니다.")
    # 받은 데이터의 합은
    total_sum = torch.zeros(1, device='cuda')
    for i in range(1, size):
        total_sum += torch.sum(received_tensors[i])
    print(f"[Rank {rank} / 서버]: 모든 클라이언트로부터 받은 텐서의 합: {total_sum.item()}")

# --- 클라이언트의 역할 ---
else:
    # 1. 사용할 GPU 장치 설정
    local_rank = rank - 1 
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise SystemError("GPU가 없습니다!")
    device = torch.device(f"cuda:{local_rank % num_gpus}")

    # 2. GPU에 텐서 데이터 생성
    tensor_to_send = torch.ones(10, device=device) * rank
    print(f"[Rank {rank} / 클라이언트]: '{device}'에서 텐서를 생성하여 보냅니다 -> {tensor_to_send}")

    # 3. GPU 텐서를 직접 전송
    req = comm.isend(tensor_to_send, dest=0, tag=11)
    req.wait()