import numpy as np 
from numpy.linalg import eig, inv
import matplotlib.pyplot as plt
import cv2
import pickle
from database_parser import database_parser, Cell
from scipy.optimize import minimize
from dataclasses import dataclass

@dataclass
class Point:
    def __init__(self,p:float,q:float,u1:float,u2:float,dist:float,G:float,sign:int) -> None:
        self.p = p
        self.q = q
        self.u1 = u1
        self.u2 = u2
        self.dist = dist
        self.G = G
        self.sign = sign
        

    def __gt__(self, other) -> bool:
        return self.u1 > other.u1

    def __lt__(self, other) -> bool:
        return self.u1 < other.u1

    def __repr__(self) -> str:
        return f"({self.u1},{self.G})"


def find_minimum_distance_and_point(coefficients, x_Q, y_Q):
        # 関数の定義
        def f_x(x):
            return sum(
                coefficient * x**i for i, coefficient in enumerate(coefficients[::-1])
            )

        # 点Qから関数上の点までの距離 D の定義
        def distance(x):
            return np.sqrt((x - x_Q) ** 2 + (f_x(x) - y_Q) ** 2)

        # scipyのminimize関数を使用して最短距離を見つける
        # 初期値は0とし、精度は低く設定して計算速度を向上させる
        result = minimize(
            distance, 0, method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-2}
        )

        # 最短距離とその時の関数上の点
        x_min = result.x[0]
        min_distance = distance(x_min)
        min_point = (x_min, f_x(x_min))

        return min_distance, min_point

@staticmethod
def find_path(
    image_fluo_raw: bytes, contour_raw: bytes, degree: int
):

    image_fluo = cv2.imdecode(
        np.frombuffer(image_fluo_raw, np.uint8), cv2.IMREAD_COLOR
    )
    image_fluo_gray = cv2.cvtColor(image_fluo, cv2.COLOR_BGR2GRAY)

    mask = np.zeros_like(image_fluo_gray)

    unpickled_contour = pickle.loads(contour_raw)
    cv2.fillPoly(mask, [unpickled_contour], 255)

    coords_inside_cell_1 = np.column_stack(np.where(mask))
    points_inside_cell_1 = image_fluo_gray[
        coords_inside_cell_1[:, 0], coords_inside_cell_1[:, 1]
    ]

    X = np.array(
        [
            [i[1] for i in coords_inside_cell_1],
            [i[0] for i in coords_inside_cell_1],
        ]
    )

    (
        u1,
        u2,
        u1_contour,
        u2_contour,
        min_u1,
        max_u1,
        u1_c,
        u2_c,
        U,
        contour_U,
    ) = basis_conversion(
        [list(i[0]) for i in unpickled_contour],
        X,
        image_fluo.shape[0] / 2,
        image_fluo.shape[1] / 2,
        coords_inside_cell_1,
    )

    theta = poly_fit(U, degree=degree)
    raw_points: list[Point] = []
    for i, j, p in zip(u1, u2, points_inside_cell_1):
        min_distance, min_point = find_minimum_distance_and_point(
            theta, i, j
        )
        sign = 1 if j > min_point[1] else -1
        raw_points.append(Point(min_point[0], min_point[1], i, j,min_distance, p,sign ))
    raw_points.sort()

    fig = plt.figure(figsize=(6, 6))
    # plot points 
    for i in raw_points:
        print(i.q,i.dist)
        plt.scatter(i.p, i.dist*i.sign, s=1,color="blue")
    # margin_width = 50
    # margin_height = 50
    # plt.xlim([min_u1 - margin_width, max_u1 + margin_width])
    # plt.ylim([min(u2) - margin_height, max(u2) + margin_height])

    fig.savefig("experimental/DotPatternMap/images/points.png")


def poly_fit(U: list[list[float]], degree: int = 1) -> list[float]:
    u1_values = np.array([i[1] for i in U])
    f_values = np.array([i[0] for i in U])
    W = np.vander(u1_values, degree + 1)
    return inv(W.T @ W) @ W.T @ f_values


def basis_conversion(
        contour: list[list[int]],
        X: np.ndarray,
        center_x: float,
        center_y: float,
        coordinates_incide_cell: list[list[int]],
    ) -> list[list[float]]:
        Sigma = np.cov(X)
        eigenvalues, eigenvectors = eig(Sigma)
        if eigenvalues[1] < eigenvalues[0]:
            Q = np.array([eigenvectors[1], eigenvectors[0]])
            U = [Q.transpose() @ np.array([i, j]) for i, j in coordinates_incide_cell]
            U = [[j, i] for i, j in U]
            contour_U = [Q.transpose() @ np.array([j, i]) for i, j in contour]
            contour_U = [[j, i] for i, j in contour_U]
            center = [center_x, center_y]
            u1_c, u2_c = center @ Q
        else:
            Q = np.array([eigenvectors[0], eigenvectors[1]])
            U = [
                Q.transpose() @ np.array([j, i]).transpose()
                for i, j in coordinates_incide_cell
            ]
            contour_U = [Q.transpose() @ np.array([i, j]) for i, j in contour]
            center = [center_x, center_y]
            u2_c, u1_c = center @ Q

        u1 = [i[1] for i in U]
        u2 = [i[0] for i in U]
        u1_contour = [i[1] for i in contour_U]
        u2_contour = [i[0] for i in contour_U]
        min_u1, max_u1 = min(u1), max(u1)
        return u1, u2, u1_contour, u2_contour, min_u1, max_u1, u1_c, u2_c, U, contour_U

def replot(
        image_fluo_raw: bytes,
        contour_raw: bytes,
        degree: int,
    ) :

        image_fluo = cv2.imdecode(
            np.frombuffer(image_fluo_raw, np.uint8), cv2.IMREAD_COLOR
        )
        image_fluo_gray = cv2.cvtColor(image_fluo, cv2.COLOR_BGR2GRAY)

        mask = np.zeros_like(image_fluo_gray)

        unpickled_contour = pickle.loads(contour_raw)
        cv2.fillPoly(mask, [unpickled_contour], 255)

        coords_inside_cell_1 = np.column_stack(np.where(mask))
        points_inside_cell_1 = image_fluo_gray[
            coords_inside_cell_1[:, 0], coords_inside_cell_1[:, 1]
        ]

        X = np.array(
            [
                [i[1] for i in coords_inside_cell_1],
                [i[0] for i in coords_inside_cell_1],
            ]
        )

        (
            u1,
            u2,
            u1_contour,
            u2_contour,
            min_u1,
            max_u1,
            u1_c,
            u2_c,
            U,
            contour_U,
        ) = basis_conversion(
            [list(i[0]) for i in unpickled_contour],
            X,
            image_fluo.shape[0] / 2,
            image_fluo.shape[1] / 2,
            coords_inside_cell_1,
        )

        fig = plt.figure(figsize=(6, 6))
        plt.scatter(u1, u2, s=5)
        plt.scatter(u1_c, u2_c, color="red", s=100)
        plt.axis("equal")
        margin_width = 50
        margin_height = 50
        plt.scatter(
            [i[1] for i in U],
            [i[0] for i in U],
            points_inside_cell_1,
            c=points_inside_cell_1,
            cmap="inferno",
            marker="o",
        )
        plt.xlim([min_u1 - margin_width, max_u1 + margin_width])
        plt.ylim([min(u2) - margin_height, max(u2) + margin_height])


        x = np.linspace(min_u1, max_u1, 1000)
        theta = poly_fit(U, degree=degree)
        y = np.polyval(theta, x)
        plt.plot(x, y, color="red")
        plt.scatter(u1_contour, u2_contour, color="lime", s=3)
        plt.tick_params(direction="in")
        plt.grid(True)
        plt.savefig("experimental/DotPatternMap/images/contour.png")

cells: list[Cell] = database_parser("sk326Gen90min.db")
print(cells[0].cell_id)

image_fluo_raw = cells[0].img_fluo1
contour_raw = cells[0].contour

replot(image_fluo_raw, contour_raw, 4)
find_path(image_fluo_raw, contour_raw, 4)