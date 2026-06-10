import numpy as np
import os
import re
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
class TiltEstimatorPlaneFit:
    def __init__(self, grid_size=32):
        self.grid_size = grid_size

    def estimate_tilt(self, pressure_grid):
        pressure_grid = np.array(pressure_grid)

        # x, y 좌표 생성
        x = np.arange(self.grid_size)
        y = np.arange(self.grid_size)
        x, y = np.meshgrid(x, y)

        # 1D 배열로 변환
        x = x.flatten()
        y = y.flatten()
        z = pressure_grid.flatten()

        # 압력이 0인 부분 제외 (노이즈 제거)
        mask = z > 0
        x = x[mask]
        y = y[mask]
        z = z[mask]

        # 디자인 매트릭스 생성
        A = np.c_[x, y, np.ones(x.shape)]

        # 최소자승법을 사용하여 평면 피팅
        C, residuals, rank, s = np.linalg.lstsq(A, z, rcond=None)
        a, b, c = C

        # 기울기 각도 계산
        angle_x = np.degrees(np.arctan(a))
        angle_y = np.degrees(np.arctan(b))

        return {
            'tilt_angle_x': angle_x,  # 좌우 기울기 각도
            'tilt_angle_y': angle_y  # 앞뒤 기울기 각도
        }

def natural_sort_key(s):
    """
    문자열을 자연스러운 숫자 순서로 정렬하기 위한 키를 반환합니다.
    예를 들어, "file10.npy"은 "file2.npy" 다음에 오게 됩니다.
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def preprocess(pressure):
    pressure[pressure <= 15] = 0
    pressure = abs((pressure/4200.)*255.).astype(np.uint8)
    return pressure

def load_and_sum_npy_files(folder_path, n):
    """
    주어진 폴더에서 n개의 npy 파일을 로드하여 합산합니다.

    Parameters:
    - folder_path (str): npy 파일들이 있는 폴더 경로
    - n (int): 더할 npy 파일의 개수

    Returns:
    - list: 각 파일의 데이터를 합산한 리스트
    """
    file_list = sorted([f for f in os.listdir(folder_path) if f.endswith('.npy')], key=natural_sort_key)

    if len(file_list) < n:
        raise ValueError("폴더에 npy 파일이 충분하지 않습니다.")

    # 첫 번째 npy 파일을 로드하여 초기화
    i = 0
    data_list = []
    for file_name in file_list[i:i+n]:
        data = np.load(os.path.join(folder_path, file_name))
        data_list.append(data)

    return data_list


# 사용 예시
if __name__ == "__main__":
    estimator = TiltEstimatorPlaneFit()

    folder_path = './data/left'  # npy 파일이 저장된 폴더 경로로 변경하세요.
    n_files = 1  # 사용할 npy 파일의 개수
    data_array = load_and_sum_npy_files(folder_path, n_files)

    # count = 0
    # for i in range(3700):
    #     # Check if any value in the 32x32 slice is <= -1000
    #     if np.any(data_array[i] <= -1000):
    #         count += 1
    #         print(f"Slice {i} has values <= -1000")
    # # Print the total count of such slices
    # print(f"Total number of slices with values <= -1000: {count}")


    data_array = np.array(data_array)

    data_array = preprocess(data_array)

    pressure_grid = data_array

    result = estimator.estimate_tilt(pressure_grid)
    print(f"Tilt Angle X (Left/Right): {result['tilt_angle_x']:.2f} degrees")
    print(f"Tilt Angle Y (Forward/Backward): {result['tilt_angle_y']:.2f} degrees")



    pressure_data = data_array
    pressure_data = np.array(pressure_data).squeeze()
    plt.figure(figsize=(6, 6))
    plt.imshow(pressure_data, cmap='viridis', origin='lower')
    plt.colorbar(label='Pressure')
    plt.title('Pressure Distribution (Heatmap)')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.show()

    # 3D 서피스 플롯 시각화
    x = np.arange(32)
    y = np.arange(32)
    x, y = np.meshgrid(x, y)
    z = pressure_data

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(x, y, z, cmap='viridis')
    fig.colorbar(surf, label='Pressure')
    ax.set_title('Pressure Distribution (3D Surface Plot)')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Pressure')

##############
    # A = np.c_[x.flatten(), y.flatten(), np.ones(x.size)]
    # C, _, _, _ = np.linalg.lstsq(A, z.flatten(), rcond=None)
    # a, b, _ = C
    #
    # # 중심에서의 기울기 벡터 계산
    # center_x = 16  # 센서 배열의 중앙 (32x32에서 16,16 위치)
    # center_y = 16
    # tilt_vector_x = a * 10  # 기울기의 크기 조절을 위해 스케일링
    # tilt_vector_y = b * 10
    #
    # # 기울기 벡터(화살표) 추가
    # ax.quiver(center_x, center_y, np.max(z), tilt_vector_x, tilt_vector_y, 0, color='red', length=5,
    #           arrow_length_ratio=0.3)
##############
    plt.show()
