import os
import numpy as np
from scipy.signal import find_peaks, savgol_filter
import natsort
import re
# 데이터 로드 함수는 이미 제공됨
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from sklearn.cluster import KMeans
def natural_sort_key(s):
    """
    문자열을 자연스러운 숫자 순서로 정렬하기 위한 키를 반환합니다.
    예를 들어, "file10.npy"은 "file2.npy" 다음에 오게 됩니다.
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def load_and_sum_npy_files(folder_path, n, num):
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
    i = num
    data_list = []
    for file_name in file_list[i:i+n]:
        data = np.load(os.path.join(folder_path, file_name))
        data_list.append(data)

    return data_list




# 1. 전체 압력 데이터의 합을 계산하여 시계열 데이터 생성
def calculate_pressure_series(data_array):
    """
    32x32 압력 데이터의 합계를 계산하여 시계열 데이터를 생성합니다.

    Parameters:
    - data_list (list): 32x32 압력 데이터의 리스트

    Returns:
    - list: 시간에 따른 압력 합계 시계열 데이터
    """
    pressure_sum = np.sum(data_array, axis=(1, 2))

    return pressure_sum

# 전체 프로세스
def measure_leg_swing_speed(folder_path, n_files, fps=12):
    # Step 1: Load data
    for i in range (0,1):
        data_array = load_and_sum_npy_files(folder_path, n_files, 3150+i)

        im_speed = np.array(data_array)

        im_speed[im_speed < 15] = 0
        im_speed = ((im_speed / 4200.) * 255).astype(np.uint8)
        #im_speed = median_filter(im_speed, size=3)
        p_min, P_max = im_speed.min(), im_speed.max()
        if P_max != p_min:
            im_speed = (im_speed - p_min) / (P_max - p_min)
        else:
            im_speed = np.zeros_like(im_speed)
        im_speed_og = im_speed

        #im_speed = im_speed[:, :, :10]
        #im_speed = im_speed[:, :10, :]
        im_speed[:, 7, :] = 0
        im_for_center = im_speed[:, :, :10]
        summed_image = np.sum(im_for_center, axis=0)
        x, y = np.meshgrid(np.arange(summed_image.shape[0]), np.arange(summed_image.shape[1]))

        non_zero_mask = summed_image > 0
        coordinates = np.column_stack(np.nonzero(non_zero_mask))
        values = summed_image[non_zero_mask]
        #coordinates = np.column_stack((x.ravel(), y.ravel()))
        #values = summed_image.ravel()
        data_for_clustering = np.column_stack((coordinates, values))
        kmeans = KMeans(n_clusters=2, random_state=0).fit(coordinates, sample_weight=values)
        cluster_centers = kmeans.cluster_centers_
        left_leg_center, right_leg_center = cluster_centers
        left_leg_amplitude = calculate_amplitude(left_leg_center, im_speed)
        right_leg_amplitude = calculate_amplitude(right_leg_center, im_speed)
        current_speed = (left_leg_amplitude + right_leg_amplitude) / 2
        current_speed = current_speed * 1.3  # 1.2
        # if current_speed <= 0.05:
        #     current_speed = 0
        # elif current_speed <= 0.5:
        #     current_speed = 0.5
        # smoothed_speed = smooth_speed(previous_speed, current_speed, alpha=0.4)
        #
        # # Step 4: Find Peaks
        # peaks = find_peaks_in_data(smoothed_series)
        #
        # # Step 5: Calculate Speed
        # speed = calculate_speed(peaks, fps)

        # Output the results
        print(f"Leg swing speed: {current_speed} movements per second")

        plt.figure(figsize=(6, 6))

        # 압력 데이터 히트맵 시각화
        rotated_image = np.rot90(im_speed_og[19], k=-1)
        flip_image = np.fliplr(rotated_image)
        plt.imshow(flip_image, cmap='viridis', origin='lower',vmin=0, vmax=1)
        plt.colorbar(label='Pressure')

        # 클러스터 중심점 시각화
        #plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', s=200, marker='x', label='Cluster Centers')
        plt.scatter(cluster_centers[0][0], cluster_centers[0][1], c='red', s=200, marker='x', label='Cluster Centers')
        plt.scatter(cluster_centers[1][0], cluster_centers[1][1], c='red', s=200, marker='x')
        # 그래프 제목과 축 레이블 설정
        plt.title('Pressure Distribution with Cluster Centers')
        plt.xlabel('left/right')
        plt.ylabel('forward/backward')

        # 레이아웃 정리 및 클러스터 중심점 표시
        plt.legend()

        # 파일로 저장
        plt.tight_layout()
        plt.savefig('pressure_with_clusters_run3.png', dpi=500, bbox_inches='tight')

        # 결과 출력
        plt.show()


def smooth_speed(previous_speed, current_speed, alpha=0.2):
    """지수 가중 이동 평균을 사용하여 속도를 스무딩."""
    smoothed_speed = alpha * current_speed + (1 - alpha) * previous_speed
    return smoothed_speed


def find_rows_with_two_chunks(data):
    rows_with_two_chunks = []

    for i, row in enumerate(data):
        # 덩어리를 찾기 위한 로직
        is_one = row > 0
        chunks = np.diff(np.concatenate(([0], is_one, [0])))
        chunk_count = np.sum(chunks == 1)

        if chunk_count == 2:  # 덩어리가 2개인 경우
            rows_with_two_chunks.append(i)

    return rows_with_two_chunks

def calculate_amplitude(center, image, radius=1):
    x, y = int(center[0]), int(center[1])
    sum_values_per_frame = []

    for t in range(image.shape[0]):
        local_values = image[t, max(0, x - radius):min(image.shape[1], x + radius + 1),
                       max(0, y - radius):min(image.shape[2], y + radius + 1)]
        # 반경 내의 값을 합산
        summed_value = np.sum(local_values)
        sum_values_per_frame.append(summed_value)

    # 최소값과 최대값 계산
    min_value = np.min(sum_values_per_frame)
    max_value = np.max(sum_values_per_frame)

    # 최소값과 최대값의 차이 계산
    amplitude = max_value - min_value

    plt.figure(figsize=(8, 6))
    plt.plot(sum_values_per_frame, marker='o', linestyle='-', color='b')
    plt.title('Pressure change around clustering point', fontsize=16)
    plt.xlabel('Frame', fontsize=12)
    plt.ylabel('Pressure', fontsize=12)
    plt.ylim(1, 3.5)
    plt.grid(True)
    plt.show()

    return amplitude


# 예시 실행
folder_path = './data/walk'  # npy 파일이 저장된 폴더 경로로 변경하세요.
n_files = 50  # 사용할 npy 파일의 개수
fps = 12
measure_leg_swing_speed(folder_path, n_files, fps)
