import numpy as np
import matplotlib.pyplot as plt
import math


def task4(vector0: np.ndarray, vector1: np.ndarray, count_features=1):
    X0 = vector0
    X1 = vector1
    distance_arr = []
    divergence_arr = []
    for phi in np.arange(0, 360, 1):
        A = np.array([
            [np.cos(phi * np.pi / 180), np.sin(phi * np.pi / 180)],
            [-np.sin(phi * np.pi / 180), np.cos(phi * np.pi / 180)]
        ])
        A = A[:, :count_features]
        X0_rotated = X0 @ A
        X0_shorted = X0_rotated[:, :count_features]

        X1_rotated = X1 @ A
        X1_shorted = X1_rotated[:, :count_features]

        B0 = np.cov(X0_shorted, rowvar=False).reshape((count_features, count_features))
        B1 = np.cov(X1_shorted, rowvar=False).reshape((count_features, count_features))

        M0 = X0_shorted.mean(axis=0)
        M1 = X1_shorted.mean(axis=0)
        distance = 1 / 8 * (M0 - M1).T @ np.linalg.inv((B0 + B1) / 2) @ (M0 - M1) + 1 / 2 * np.log(
            ((1 / 2) * np.linalg.det(B0 + B1)) / np.linalg.det(B0) ** (1 / 2) * np.linalg.det(B1) ** (1 / 2))
        distance_arr.append(distance)

        divergence = (1 / 2 * (M0 - M1) @ (np.linalg.inv(B0) + np.linalg.inv(B1)) @ (M0 - M1).T + 1 / 2 *
                      (np.linalg.inv(B0) @ B1 + np.linalg.inv(B1) @ B0 - 2 * np.eye(count_features)).trace())
        divergence_arr.append(divergence)

    distance_arr = np.array(distance_arr)
    divergence_arr = np.array(divergence_arr)

    print("\nЗадание 4:")
    print(f"bhattacharya (max={distance_arr.max()}) matrix, phi={distance_arr.argmax()}")
    print(np.asarray([[np.cos(np.radians(distance_arr.argmax())), 0],
                      [-np.sin(np.radians(distance_arr.argmax())), 0]]))
    print(f"divergence (max={divergence_arr.max()}) matrix, phi={divergence_arr.argmax()}")
    print(np.asarray([[np.cos(np.radians(divergence_arr.argmax())), 0],
                      [-np.sin(np.radians(divergence_arr.argmax())), 0]]))
    plt.plot(np.arange(0, 360, 1), distance_arr, label='bhattacharya')
    plt.plot(distance_arr.argmax(), distance_arr.max(), 'x', color='red')
    plt.plot(np.arange(0, 360, 1), divergence_arr, label='divergence')
    plt.plot(divergence_arr.argmax(), divergence_arr.max(), 'x', color='red')
    plt.legend()
    plt.show()


def generate_random_vectors(mean_vector: np.ndarray, cov_matrix: np.ndarray, N: int = 200, ) -> (
        np.ndarray, np.ndarray):
    sequences = np.random.multivariate_normal(mean_vector, cov_matrix, N)
    return sequences


def Carunen_Loev(X: np.ndarray, B: np.ndarray):
    B = np.cov(X, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(B)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    W = eigenvalues[sorted_indices]
    A = eigenvectors[:, sorted_indices]
    A[0][1] = 0
    A[1][1] = 0

    X_centered = X - np.mean(X, axis=0)
    Y = X_centered @ A
    X_hat = Y @ A.T
    # plt.plot(X_hat[:, 0], X_hat[:, 1], '.', alpha=0.5)
    # plt.plot(X[:, 0], X[:, 1], '.', alpha=0.5)
    # plt.show()
    emper_error = (np.sum(np.linalg.norm(X_hat - X_centered) ** 2)) / X_centered.shape[0]
    theor_error = np.sum(W[1:])
    print("\nЗадание 2")
    print(f'Эмпирическая ошибка = {emper_error}')
    print(f'Теоритическая ошибка = {theor_error}')
    print(f'Собственные значения  = {W}')

    phi_values = np.arange(0, 360, 1)
    # err_values = np.array(phi_values.size)
    err_values = []
    for phi in phi_values:
        A = np.array([[np.cos(phi * np.pi / 180), np.sin(phi * np.pi / 180)],
                      [-np.sin(phi * np.pi / 180), np.cos(phi * np.pi / 180)]])
        A[0][1] = 0
        A[1][1] = 0

        Y = X_centered @ A
        X_hat = Y @ A.T

        emper_error = (np.sum(np.linalg.norm(X_hat - X_centered) ** 2)) / X_centered.shape[0]

        err_values.append(emper_error)

    err_values_arr = np.array(err_values)
    plt.plot(phi_values, err_values_arr)
    plt.plot(err_values_arr.argmin(), err_values_arr.min(), 'x', color='black')

    plt.show()

    A = np.array([
        [np.cos(err_values_arr.argmin() * np.pi / 180), np.sin(err_values_arr.argmin() * np.pi / 180)],
        [-np.sin(err_values_arr.argmin() * np.pi / 180), np.cos(err_values_arr.argmin() * np.pi / 180)]
    ])
    print(f"Матрица A при phi = {err_values_arr.argmin()} и наименьшей ошибкой = {err_values_arr.min()}")
    print(A)
    A[0][1] = 0
    A[1][1] = 0

    Y = X_centered @ A
    X_hat = Y @ A.T

    return X_hat, eigenvectors


def task3(vector0: np.ndarray, vector1: np.ndarray):
    M0 = vector0.mean(axis=0).reshape([1, vector0.shape[1]])
    M1 = vector1.mean(axis=0).reshape([1, vector1.shape[1]])

    MS = (0.5 * M0) + (0.5 * M1)

    # J1
    S1 = 0.5 * (M0 - MS).T @ (M0 - MS) + 0.5 * (M1 - MS).T @ (M1 - MS)  # S1 = Bb1

    B0 = np.cov(vector0, rowvar=False)
    B1 = np.cov(vector1, rowvar=False)

    S2 = 0.5 * B0 + 0.5 * B1  # S2 = B_summ
    temp_matrix = np.linalg.inv(S2) @ S1

    eigenvalues, eigenvectors = np.linalg.eigh(temp_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]

    # print(eigenvalues)

    A = eigenvectors[:, sorted_indices]
    A[0][1] = 0
    A[1][1] = 0
    # print(A)

    Y0 = vector0 @ A
    Y1 = vector1 @ A

    X0_new = Y0 @ A.T
    X1_new = Y1 @ A.T

    plt.plot(X0_new[:, 0], X0_new[:, 1], '.', alpha=0.5)
    plt.plot(X1_new[:, 0], X1_new[:, 1], '.', alpha=0.5)
    plt.plot(vector0[:, 0], vector0[:, 1], '.', alpha=0.5)
    plt.plot(vector1[:, 0], vector1[:, 1], '.', alpha=0.5)
    plt.show()

    # J2
    S1 = (0.5 * B0 + 0.5 * B1) + 0.5 * (M0 - MS).T @ (M0 - MS) + 0.5 * (M1 - MS).T @ (M1 - MS)  # S1 = B_summ + Bb1
    S2 = 0.5 * B0 + 0.5 * B1  # S1 = B_summ
    temp_matrix = np.linalg.inv(S2) @ S1

    eigenvalues, eigenvectors = np.linalg.eigh(temp_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]

    A = eigenvectors[:, sorted_indices]
    # A = A[:, :1]
    A[0][1] = 0
    A[1][1] = 0
    # print(A)
    Y0 = vector0 @ A
    Y1 = vector1 @ A

    X0_new = Y0 @ A.T
    X1_new = Y1 @ A.T

    plt.plot(X0_new[:, 0], X0_new[:, 1], '.', alpha=0.5)
    plt.plot(X1_new[:, 0], X1_new[:, 1], '.', alpha=0.5)
    plt.plot(vector0[:, 0], vector0[:, 1], '.', alpha=0.5)
    plt.plot(vector1[:, 0], vector1[:, 1], '.', alpha=0.5)
    plt.show()

    J1 = []
    J2 = []
    for phi in np.arange(0, 360, 1):
        A = np.array([
            [np.cos(phi * np.pi / 180), np.sin(phi * np.pi / 180)],
            [-np.sin(phi * np.pi / 180), np.cos(phi * np.pi / 180)]
        ])
        A = A[:, :1]

        X0_hat = vector0 @ A
        X1_hat = vector1 @ A

        M0 = X0_hat.mean(axis=0).reshape([1, X0_hat.shape[1]])
        M1 = X1_hat.mean(axis=0).reshape([1, X1_hat.shape[1]])
        MS = (0.5 * M0) + (0.5 * M1)
        S1 = 0.5 * (M0 - MS).T @ (M0 - MS) + 0.5 * (M1 - MS).T @ (M1 - MS)
        cov0 = np.cov(X0_hat, rowvar=False).reshape([1, 1])
        cov1 = np.cov(X1_hat, rowvar=False).reshape([1, 1])
        S2 = 0.5 * cov0 + 0.5 * cov1  # S2 = B_summ
        temp_matrix = np.linalg.inv(S2) @ S1
        J1.append(temp_matrix.trace())
        J2.append(np.log(np.linalg.det(temp_matrix)))

    J1 = np.array(J1)
    J2 = np.array(J2)
    print("\nЗадание 3:")
    print(f"Лучшая J1 = {J1.max()}, Лучшая J2 = {J2.max()}")
    A = np.array([
        [np.cos(J1.argmax() * np.pi / 180), np.sin(J1.argmax() * np.pi / 180)],
        [-np.sin(J1.argmax() * np.pi / 180), np.cos(J1.argmax() * np.pi / 180)]
    ])
    print(f"Лучшая A = \n{A}")
    A[0][1] = 0
    A[1][1] = 0

    plt.plot(np.arange(0, 360, 1), J1, label='J1')
    plt.plot(np.arange(0, 360, 1), J2, label='J2')
    plt.plot(J1.argmax(), J1.max(), 'x', color='red')
    plt.plot(J2.argmax(), J2.max(), 'x', color='red')
    plt.legend()
    plt.show()

    Y0 = vector0 @ A
    Y1 = vector1 @ A

    X0_hat = Y0 @ A.T
    X1_hat = Y1 @ A.T

    return X0_hat, X1_hat


def main():
    np.random.seed(52)
    M1 = np.array((0, 0))
    M2 = np.array((1, -1))
    M3 = np.array((-1, -1))
    D1 = np.array([0.4, 0.3])
    D2 = np.array([0.1, 0.1])
    D3 = np.array([0.1, 0.1])

    rho = 0.5

    cov_matrix1 = np.array([
        [D1[0], rho * math.sqrt(D1[0]) * math.sqrt(D1[1])],
        [rho * math.sqrt(D1[0]) * math.sqrt(D1[1]), D1[1]]
    ])
    cov_matrix2 = np.array([
        [D2[0], rho * math.sqrt(D2[0]) * math.sqrt(D2[1])],
        [rho * math.sqrt(D2[0]) * math.sqrt(D2[1]), D2[1]]
    ])
    cov_matrix3 = np.array([
        [D3[0], rho * math.sqrt(D3[0]) * math.sqrt(D3[1])],
        [rho * math.sqrt(D3[0]) * math.sqrt(D3[1]), D3[1]]
    ])

    rand_vect1 = generate_random_vectors(M1, cov_matrix1)
    rand_vect2 = generate_random_vectors(M2, cov_matrix2)

    plt.scatter(rand_vect1[:, 0], rand_vect1[:, 1], marker='.')
    plt.scatter(rand_vect2[:, 0], rand_vect2[:, 1], marker='.')

    plt.show()

    V0, _ = Carunen_Loev(rand_vect1, cov_matrix1)
    plt.scatter(rand_vect1[:, 0], rand_vect1[:, 1], marker='.', alpha=0.5)
    plt.scatter(rand_vect2[:, 0], rand_vect2[:, 1], marker='.', alpha=0.5)
    plt.scatter(V0[:, 0], V0[:, 1], marker='.')
    plt.show()

    V0, V1 = task3(rand_vect1, rand_vect2)
    plt.scatter(rand_vect1[:, 0], rand_vect1[:, 1], marker='.', alpha=0.5)
    plt.scatter(rand_vect2[:, 0], rand_vect2[:, 1], marker='.', alpha=0.5)
    plt.scatter(V0[:, 0], V0[:, 1], marker='.')
    plt.scatter(V1[:, 0], V1[:, 1], marker='.')
    plt.show()


if __name__ == "__main__":
    main()
