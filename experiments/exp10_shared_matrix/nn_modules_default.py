"""Учебная копия nn_modules.py: только default-архитектура LISA.

Источник: code/LISA/src/lisa/model/nn_modules.py.
Default-размеры взяты из parse_experiment_args.py и utils/constants.py.
Удалены альтернативные режимы, анализ фильтров и абстрактный класс внимания.
В копии исправлена локальная опечатка kernel_sie -> kernel_size.
Это отдельный файл для чтения, основной проект его не использует.
"""

# Читать удобно с класса LISA внизу, переходя к вызываемым слоям через F12.
# B — размер батча, T — число отсчётов времени.
# Путь: 208 датчиков -> 270 -> 270 -> 25 -> фильтр -> два блока -> 768 признаков.
# Например, T=300 (3 секунды при 100 Гц) -> 149 позиций после головы.
# Координаты coords_xyz нужно передать из sensor_xyz.npy, форма (208, 3).
# seed управляет только Z внимания; общий seed задаётся до создания модели
# через torch.manual_seed(42), как это делает обучение автора.

import math

import einops
import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F


def cart2sph(sensor_xyz):
    """Координаты датчиков (x, y, z) -> (радиус, theta, phi)."""
    x, y, z = sensor_xyz[:, 0], sensor_xyz[:, 1], sensor_xyz[:, 2]
    xy = np.linalg.norm(sensor_xyz[:, :2], axis=-1)
    r = np.linalg.norm(sensor_xyz, axis=-1)
    theta = np.arctan2(xy, z)
    phi = np.arctan2(y, x)
    return np.stack((r, theta, phi), axis=-1)


class Spatial3DAttentionLayer(nn.Module):
    """Смешивание 208 датчиков в 270 каналов с весами из сферических гармоник."""

    def __init__(
        self,
        coords_xyz,  # Положение датчиков: (208, 3).
        seed=42,  # Seed начальных обучаемых коэффициентов Z.
    ):
        super().__init__()
        self.n_input = 208
        self.n_output = 270
        self.K = 24  # Степени гармоник 0..23, всего 24*24 функций.

        # Фиксированный базис в точках датчиков: (1, 24, 24, 208).
        coords_sph = cart2sph(coords_xyz)
        layout = self._create_layout(coords_sph, self.K - 1)
        self.register_buffer('_layout', layout)

        # Обучаемая комбинация гармоник для каждого из 270 выходных каналов.
        generator = torch.Generator()
        generator.manual_seed(seed)
        Z = (
            torch.randn(size=(self.n_output, self.K, self.K), generator=generator)
            * 2
            / (self.n_input + self.n_output)
        )
        # (270, 24, 24) -> (270, 24, 24, 1), чтобы умножать на layout.
        self.Z = nn.Parameter(Z.unsqueeze(-1))

    def _create_layout(self, coords_sph, L=8):
        """Расчёт базиса из оригинала; вызывается только при создании слоя."""
        n_input = coords_sph.shape[0]

        coords_theta = coords_sph[:, 1].astype(np.float64)
        coords_phi = coords_sph[:, 2].astype(np.float32)

        # Фиксированные значения функций Лежандра для углов датчиков.
        res = scipy.special.assoc_legendre_p_all(
            L, L, np.cos(coords_theta), norm=False, diff_n=0
        )
        assert res.shape == (1, L + 1, 2 * L + 1, n_input)

        Plak_values = res[0, :, : L + 1, :].transpose(1, 0, 2)

        def get_factor(l, m):
            # Нормировка гармоники степени l и порядка m.
            result = math.log(2 * l + 1)
            result -= math.log((2 if m == 0 else 1) * 2 * math.pi)
            result += scipy.special.gammaln(l - abs(m) + 1)
            result -= scipy.special.gammaln(l + abs(m) + 1)
            result /= 2
            result = math.exp(result)
            return result

        layout = np.zeros((1, L + 1, L + 1, n_input), dtype=np.float32)
        counter = -1
        for l in range(0, L + 1):
            for m in range(-l, l + 1):
                counter += 1
                i, j = counter % (L + 1), counter // (L + 1)
                mult_left = get_factor(l, m)

                if m >= 0:
                    mult = np.cos(m * coords_phi)
                elif m < 0:
                    mult = np.sin(-m * coords_phi)
                sela = mult_left * Plak_values[abs(m), l, :] * mult

                layout[:, i, j, :] = sela
        layout = torch.tensor(layout, dtype=torch.float32, requires_grad=False)
        return layout

    def _get_attention_matrix(self):
        # Обычный PyTorch: A = (self.Z * self._layout).sum(dim=(1, 2))
        # (270, 24, 24, 1) * (1, 24, 24, 208) -> сумма -> (270, 208).
        A = einops.reduce(self.Z * self._layout, 'j k l i -> j i', 'sum')
        return A

    def forward(self, x):
        # x: (B, 208, T). Матрица зависит от Z и геометрии, а не от текущего x.
        A = self._get_attention_matrix()
        ASoftmax = F.softmax(A, dim=1)  # Сумма весов датчиков в каждой строке = 1.

        # Обычный PyTorch: SAx = torch.matmul(ASoftmax, x)
        # (270, 208) @ (B, 208, T) -> (B, 270, T).
        SAx = torch.einsum('oi, bit -> bot', ASoftmax, x)
        return SAx


class SubjectPlusLayer(nn.Module):
    """У каждого участника своя матрица: 270 входных каналов -> 25 выходных."""

    def __init__(self, n_input, n_output, n_subjects):
        # В default: n_input=270, n_output=25, n_subjects=27.
        super().__init__()
        A = self._create_parameters(n_input, n_output, n_subjects)
        self.A = nn.Parameter(A)

    def _create_parameters(self, n_input, n_output, n_subjects):
        A = torch.zeros(size=(n_subjects, n_output, n_input))
        with torch.no_grad():
            for subjects in range(n_subjects):
                # Как у автора: используем стандартную инициализацию Conv1d.
                layer = nn.Conv1d(
                    in_channels=n_input, out_channels=n_output, kernel_size=1
                )
                # Обычный PyTorch: layer.weight.data.squeeze(-1)
                A[subjects] = einops.rearrange(layer.weight.data, 'o i 1 -> o i')
        return A

    def forward(self, x, s):
        # x: (B, 270, T); s: (B,), ID участников 0..26.
        assert s.dtype == torch.long
        if int(s.min()) < 0 or int(s.max()) >= self.A.size(0):
            raise ValueError('Subject indices must be in [0, 26].')

        A_ = self.A[s, :, :]  # Банк (27, 25, 270) -> матрицы батча (B, 25, 270).
        # Обычный PyTorch: out = torch.bmm(A_, x)
        # (B, 25, 270) @ (B, 270, T) -> (B, 25, T).
        out = torch.einsum('bji, bit -> bjt', A_, x)
        return out


class ConvBlock(nn.Module):
    """Один временной блок: свёртки, остаточные связи, BatchNorm, GELU и GLU."""

    def __init__(self, n_input, n_output, block_index):
        # В default: оба размера равны 25; block_index равен 0 или 1.
        super().__init__()
        self.kernel_size = 3
        self.block_index = block_index

        # Dilation — расстояние между точками ядра.
        # Блок 0: (1, 2, 2). Блок 1: (4, 8, 2).
        dilation1 = 2 ** (2 * block_index % 5)
        dilation2 = 2 ** ((2 * block_index + 1) % 5)
        dilation3 = 2

        self.conv1 = nn.Conv1d(
            in_channels=n_input,
            out_channels=n_output,
            kernel_size=self.kernel_size,
            dilation=dilation1,
            padding='same',
        )
        self.conv2 = nn.Conv1d(
            in_channels=n_output,
            out_channels=n_output,
            kernel_size=self.kernel_size,
            dilation=dilation2,
            padding='same',
        )
        self.conv3 = nn.Conv1d(
            in_channels=n_output,
            out_channels=2 * n_output,  # 25 -> 50 каналов перед GLU.
            kernel_size=self.kernel_size,
            dilation=dilation3,
            padding='same',
        )
        self.batchnorm1 = nn.BatchNorm1d(n_output)
        self.batchnorm2 = nn.BatchNorm1d(n_output)
        self.activation1 = nn.GELU()
        self.activation2 = nn.GELU()
        self.activation3 = nn.GLU(dim=-2)

    def forward(self, x):
        # Первая свёртка; во втором блоке добавляем её вход (остаточная связь).
        c1x = self.conv1(x)
        res1 = c1x if self.block_index == 0 else x + c1x
        res1 = self.batchnorm1(res1)
        res1 = self.activation1(res1)

        # Вторая свёртка с остаточной связью в обоих блоках.
        c2x = self.conv2(res1)
        res2 = res1 + c2x
        res2 = self.batchnorm2(res2)
        res2 = self.activation2(res2)

        c3x = self.conv3(res2)  # (B, 25, T) -> (B, 50, T).
        # GLU: делим на две половины, возвращаем a * sigmoid(b).
        out = self.activation3(c3x)  # (B, 50, T) -> (B, 25, T).
        return out


class ConvHead(nn.Module):
    """Только default-голова single_conv: 25 каналов -> 768 признаков."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=25,
            out_channels=768,
            kernel_size=3,  # В оригинальной локальной копии здесь опечатка kernel_sie.
            stride=2,
            padding=0,
        )
        self.activation = nn.GELU()
        self.batch_norm = nn.BatchNorm1d(768)

    def forward(self, x):
        # У автора GELU стоит ПЕРЕД свёрткой.
        x = self.activation(x)
        x = self.conv(x)  # (B, 25, T) -> (B, 768, (T - 3) // 2 + 1).
        x = self.batch_norm(x)
        return x


class SpatialModule(nn.Module):
    """Внимание -> общая проекция -> матрица участника -> временной фильтр."""

    def __init__(self, coords_xyz, seed=42):
        super().__init__()
        self.self_attention = Spatial3DAttentionLayer(coords_xyz, seed=seed)
        self.unmixing_layer = nn.Conv1d(
            in_channels=270,
            out_channels=270,
            kernel_size=1,  # Смешивание каналов отдельно в каждый момент времени.
            bias=True,
        )
        self.subject_layer = SubjectPlusLayer(
            n_input=270,
            n_output=25,
            n_subjects=27,
        )
        self.temporal_filter = nn.Conv1d(
            in_channels=25,
            out_channels=25,
            kernel_size=15,  # 150 мс при частоте дискретизации 100 Гц.
            padding='same',
            padding_mode='zeros',
            groups=25,  # Отдельное временное ядро для каждого канала.
            bias=False,
        )
        # FIR-полос нет: веса фильтра случайные и обучаются с первой эпохи.
        # Dropout, gate и GELU этой части в default выключены.

    def forward(self, xs):
        x, s = xs  # MEG: (B, 208, T), ID участников: (B,).
        x = self.self_attention(x)  # -> (B, 270, T).
        x = self.unmixing_layer(x)  # -> (B, 270, T).
        x = self.subject_layer(x, s)  # -> (B, 25, T).
        x = self.temporal_filter(x)  # -> (B, 25, T).
        return x  # Интерпретируемое представление Z1.


class TemporalModule(nn.Module):
    """Два последовательных временных блока, каждый сохраняет (B, 25, T)."""

    def __init__(self):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            ConvBlock(n_input=25, n_output=25, block_index=0),
            ConvBlock(n_input=25, n_output=25, block_index=1),
        )

    def forward(self, x):
        return self.conv_blocks(x)


class LISA(nn.Module):
    """Вся default-модель. Начни чтение с forward этого класса."""

    def __init__(self, coords_xyz, seed=42):
        super().__init__()
        self.spatial_module = SpatialModule(coords_xyz, seed=seed)
        self.temporal_module = TemporalModule()
        self.feature_projection = ConvHead()

    def forward_interpretable(self, xs):
        # xs = (meg, subject_idx) -> Z1: (B, 25, T).
        return self.spatial_module(xs)

    def forward_tail(self, z):
        h = self.temporal_module(z)  # (B, 25, T).
        y = self.feature_projection(h)  # (B, 768, (T - 3) // 2 + 1).
        return y

    def forward(self, xs):
        z = self.forward_interpretable(xs)
        y = self.forward_tail(z)
        return z, y  # При T=300: (B, 25, 300) и (B, 768, 149).
