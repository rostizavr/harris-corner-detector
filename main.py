import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


def img_gray(img_path):
    """
    Преобразует цветное изображение в градации серого,
    используя среднее значение RGB-компонентов каждого пикселя

    :param img_path: Путь к исходному изображению
    :return gray_image: Изображение в градациях серого
    """
    image = Image.open(img_path).convert('RGB')
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for i in range(width):
        for j in range(height):
            r, g, b = image.getpixel((i, j))
            S = int((r + g + b) / 3)
            draw.point((i, j), (S, S, S))
    gray_image = image.convert('L')
    return gray_image


def my_corner_peaks(response, min_distance=1, threshold_rel=0.1):
    """
    Обнаруживает пики (углы) на изображении,
    используя заданный порог относительной интенсивности
    и минимальное расстояние между углами.

    :param response: Матрица отклика, полученная из алгоритма обнаружения углов
    :param min_distance: Минимальное расстояние между обнаруженными углами
    :param threshold_rel: Пороговое значение
    :return numpy.array(coords): Массив координат обнаруженных углов
    """
    min_distance = max(1, int(min_distance))
    threshold = np.max(response) * threshold_rel
    thresholded_response = response > threshold
    coords = []
    for row in range(min_distance, response.shape[0] - min_distance):
        for col in range(min_distance, response.shape[1] - min_distance):
            if thresholded_response[row, col]:
                local_region = response[
                    row - min_distance:row + min_distance + 1,
                    col - min_distance:col + min_distance + 1]

                if response[row, col] == np.max(local_region):
                    coords.append((row, col))

    return np.array(coords)


def corr(img, mask):
    """
    Выполняет корреляционное преобразование изображения
    с использованием заданной маски.

    :param img: Исходное изображение в виде матрицы
    :param mask: Маска фильтра
    :return filtered_img: Изображение после применения преобразования
    """
    row, col = img.shape
    m, n = mask.shape
    new = np.zeros((row+m-1, col+n-1))
    n //= 2
    m //= 2
    filtered_img = np.zeros(img.shape)
    new[m:new.shape[0]-m, n:new.shape[1]-n] = img
    for i in range(m, new.shape[0]-m):
        for j in range(n, new.shape[1]-n):
            temp = new[i-m:i+m+1, j-m:j+m+1]
            result = temp*mask
            filtered_img[i-m, j-n] = result.sum()
    return filtered_img


def gaussian(m, n, sigma):
    """
    Генерирует Гауссов фильтр заданного размера и сигмы.

    :param m: ширина фильтра
    :param n: высота фильтра
    :param sigma: Стандартное отклонение Гауссова распределения
    :return gaussian: Матрица Гауссова фильтра
    """
    gaussian = np.zeros((m, n))
    m //= 2
    n //= 2
    for x in range(-m, m+1):
        for y in range(-n, n+1):
            x1 = sigma*(2*np.pi)**2
            x2 = np.exp(-(x**2+y**2)/(2*sigma**2))
            gaussian[x+m, y+n] = (1/x1)*x2
    return gaussian


def conv2d(image, kernel):
    """
    Производит двумерную свертку изображения с ядром
    применяя фильтр Собеля,
    вычисляет пространственную производную

    :param image: Исходное изображение.
    :param kernel: Ядро (фильтр) для свертки
    :return b: Результат свертки
    """
    k = kernel.shape[0]
    width = k//2
    a = framed(image, width)
    b = np.zeros(image.shape)
    for p, dp, r, dr in \
            [(i, i + image.shape[0], j, j + image.shape[1])
             for i in range(k) for j in range(k)]:
        b += a[p:dp, r:dr] * kernel[p, r]
    return b


def framed(image, width):
    """
    Создает рамку вокруг изображения
    с помощью добавления пустых границ заданной ширины.

    :param image: Исходное изображение в виде двумерного массива
    :param width: Ширина рамки
    :return a: Новое изображение с добавленной рамкой
    """
    a = np.zeros((image.shape[0]+2*width, image.shape[1]+2*width))
    a[width:-width, width:-width] = image
    return a


def harris_corners(image, k=0.05, window_size=2, sigma=1.0):
    """
    Обнаружение углов на изображении с использованием метода Харриса

    :param image: Исходное изображение
    :param k: Параметр Харриса для вычисления отклика Харриса
    :param window_size: Размер окна для локального суммирования угловых мер
    :param sigma: Стандартное отклонение Гауссовского фильтра
    :return harris_response: Матрица откликов Харриса
    """
    kernel_size = 5
    gaussian_kernel = gaussian(kernel_size, kernel_size, sigma)

    smoothed_image = conv2d(image, gaussian_kernel)

    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    Ix = conv2d(smoothed_image, kernel_x)
    Iy = conv2d(smoothed_image, kernel_y)

    Ixx = Ix ** 2
    Iyy = Iy ** 2
    Ixy = Ix * Iy

    height, width = image.shape[:2]
    offset = window_size // 2

    harris_response = np.zeros((height, width))

    for y in range(offset, height - offset):
        for x in range(offset, width - offset):
            Sxx = np.sum(Ixx[y-offset:y+1+offset, x-offset:x+1+offset])
            Syy = np.sum(Iyy[y-offset:y+1+offset, x-offset:x+1+offset])
            Sxy = np.sum(Ixy[y-offset:y+1+offset, x-offset:x+1+offset])
            detA = Sxx * Syy - Sxy ** 2
            traceA = Sxx + Syy
            harris_response[y, x] = detA - k * traceA ** 2
    return harris_response


def recolor_red_tones_in_image(gray_image_path, color_image_path, output_path):
    """
    Перекрашивает пиксели в красные тона в цветном изображении,
    используя в качестве маски изображение в оттенках серого.
    Тоны, которые в сером изображении являются красными,
    перекрашиваются в ярко-красный цвет в цветном изображении

    :param gray_image_path: путь до изображения в оттенках серого
    :param color_image_path: путь до цветного изображения
    :param output_path: имя выходного изображения
    :return:
    """
    gray_image = Image.open(gray_image_path)
    gray_array = np.array(gray_image)

    color_image = Image.open(color_image_path)
    color_array = np.array(color_image)

    if gray_array.shape[:2] != color_array.shape[:2]:
        raise ValueError("Изображения имеют разные размеры или каналы.")

    red_tones = (gray_array[:, :, 0] > gray_array[:, :, 1]) & (
        gray_array[:, :, 0] > gray_array[:, :, 2])

    if color_array.shape[2] == 4:
        color_array[red_tones] = [255, 0, 0, 255]
    else:
        color_array[red_tones] = [255, 0, 0]

    recolored_image = Image.fromarray(color_array)
    recolored_image.save(output_path)

    return output_path


def find_corners(image, threshold=0.01):
    """
    Обертка для детектирования углов на изображении.
    Сначала применяет детектор углов Харриса к изображению,
    затем использует функцию my_corner_peaks для определения координат углов

    :param image: Изображение, представленное в виде массива
    :param threshold: пороговое значение
    :return corners: координаты обнаруженных углов
    """
    harris_response = harris_corners(image)
    corners = my_corner_peaks(harris_response, threshold_rel=threshold)
    return corners


img_path = input("Введите путь к изображению: ")
image = np.array(img_gray(img_path), dtype=np.float64)
corners = find_corners(image)
im = Image.open(img_path)

fig, ax = plt.subplots()
fig1 = plt.figure(figsize=(6.4, 4.8))
ax1 = fig1.add_subplot()
ax.imshow(image, cmap=plt.cm.gray)
ax1.imshow(im)

fig1.savefig('original.png', bbox_inches='tight', pad_inches=0)
fig.savefig('gray_image.png', bbox_inches='tight', pad_inches=0)
ax.plot(corners[:, 1], corners[:, 0], 'r.', markersize=1)
plt.axis('off')
fig.savefig('detected_corners_gray.png', bbox_inches='tight', pad_inches=0)
plt.close(fig)

gray_image_path = 'detected_corners_gray.png'
color_image_path = 'original.png'
output_path = 'detected_corners.png'
recolor_red_tones_in_image(gray_image_path, color_image_path, output_path)

corners_file_path = 'detected_corners_coordinates.txt'
with open(corners_file_path, 'w') as file:
    for corner in corners:
        file.write(f'{corner[0]}, {corner[1]}\n')
