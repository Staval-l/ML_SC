import math
import numpy as np
import matplotlib.pyplot as plt


def generate_vectors_metod(m, k, mu, d, r):
    size = m * k
    y = []
    x = np.random.normal(0, 1, size)
    y_0 = math.sqrt(d * (1 - (r ** 2))) * x[0]
    y.append(y_0)
    for i in range(1, size):
        y.append(r * y[i - 1] + math.sqrt(d * (1 - (r ** 2))) * x[i])
    res = np.array(y)
    res += mu
    return res


def task_1(m, k, m0, m1, d0, d1, r):
    seq1 = generate_vectors_metod(m, k, m0, d0, r)
    seq2 = generate_vectors_metod(m, k, m1, d1, r)
    return seq1, seq2


def task_2(seq1, seq2, m, k):
    res_1 = np.reshape(seq1, (m, k))
    res_2 = np.reshape(seq2, (m, k))
    plt.scatter(range(k), res_1[0], s=10, marker="x", c="blue", label="X0")
    plt.scatter(range(k), res_2[0], s=10, marker="o", c="red", label="X1")
    plt.legend()
    plt.show()
    return res_1, res_2


def LLR(x, m0, m1, d0, d1):
    return (1 / 2 * np.log(d0 / d1)
            + ((np.power((x - m0), 2)) / (2 * d0))
            - ((np.power((x - m1), 2)) / (2 * d1)))


def f(x, m, d, r):
    b = np.full((x.shape[0], x.shape[0]), r * d)
    np.fill_diagonal(b, d)
    p = np.transpose(x - m) @ np.linalg.inv(b.copy()) @ (x - m)
    return (
            np.exp(p / -2) *
            (1 / np.power(2 * np.pi * np.linalg.det(b), x.shape[0] / 2))
    )


def calc_OP(x, m0, m1, d0, d1, r):
    f1 = f(x, d1, m1, r)
    f2 = f(x, d0, m0, r)
    return f1 / f2


def task_3(seq, m0, m1, d0, d1, p0, p1, r):
    l_list = []
    a = np.log((1 - p1) / p0)
    b = np.log(p1 / (1 - p0))
    l = LLR(np.array(seq[0]), m0, m1, d0, d1)
    l_list.append(l)
    for i in range(1, seq.size):
        if l >= a:
            return 1, i, l_list
        if l <= b:
            return 0, i, l_list
        l += LLR(np.array(seq[i]), m0, m1, d0, d1)
        l_list.append(l)

    return np.inf, seq.size, l_list


def E_Z_Omega_1_0(m0, m1, d):
    return ((m1 - m0) ** 2) / (2 * d), (-1 * ((m1 - m0) ** 2) / (2 * d))


def calc_E_n_Omega(p0, p1, E_Z_O_1, E_Z_O_0):
    a = np.log((1 - p1) / p0)
    b = np.log(p1 / (1 - p0))
    E_n_Omega_0 = (a * p0 + b * (1 - p0)) / E_Z_O_0
    E_n_Omega_1 = (a * (1 - p1) + b * p1) / E_Z_O_1
    return E_n_Omega_0, E_n_Omega_1


def calc_teor_p(p0, p1):
    a = (1 - p1) / p0
    b = p1 / (1 - p0)
    p1 = b * (a - 1) / (a - b)
    p0 = (1 - b) / (a - b)
    return p0, p1


def task_4(seq0, seq1, m0, m1, d0, d1, p0, p1, r):
    prediction_res_0 = []
    prediction_res_1 = []
    l_list_0 = []
    prediction_iter_count_0 = []
    prediction_iter_count_1 = []
    l_list_1 = []
    a = np.log((1 - p1) / p0)
    b = np.log(p1 / (1 - p0))
    for i in range(seq0.shape[0]):
        res = task_3(np.array(seq0[i]), m0, m1, d0, d1, p0, p1, r)
        prediction_res_0.append(res[0])
        prediction_iter_count_0.append(res[1])
        l_list_0.append(res[2])
    for i in range(seq1.shape[0]):
        res = task_3(seq1[i], m0, m1, d0, d1, p0, p1, r)
        prediction_res_1.append(res[0])
        prediction_iter_count_1.append(res[1])
        l_list_1.append(res[2])
    plt.scatter(np.arange(0, prediction_iter_count_0[5]), y=l_list_0[5], marker="x", c="blue")
    plt.axhline(a, color='red', linestyle='--', linewidth=2, label='a')
    plt.axhline(b, color='red', linestyle='--', linewidth=2, label='b')
    plt.legend()
    plt.show()
    plt.scatter(np.arange(0, prediction_iter_count_1[5]), l_list_1[5], marker="x", c="blue")
    plt.axhline(a, color='red', linestyle='--', linewidth=2, label='a')
    plt.axhline(b, color='red', linestyle='--', linewidth=2, label='b')
    plt.legend()
    plt.show()
    E_Z_O_1, E_Z_O_0 = E_Z_Omega_1_0(m0, m1, d0)
    E_n_Omega_0, E_n_Omega_1 = calc_E_n_Omega(p0, p1, E_Z_O_1, E_Z_O_0)
    teor_p0, teor_p1 = calc_teor_p(p0, p1)
    print("Theoretical")
    print("E{n|Omega_0} = " + str(E_n_Omega_0))
    print("E{n|Omega_1} = " + str(E_n_Omega_1))
    print("p0 = " + str(teor_p0))
    print("p1 = " + str(teor_p1))
    print("Practical")
    print("E{n|Omega_0} = " + str(sum(prediction_iter_count_0) / len(prediction_iter_count_0)))
    print("E{n|Omega_1} = " + str(sum(prediction_iter_count_1) / len(prediction_iter_count_1)))
    print("p0 = " + str(sum(prediction_res_0) / len(seq0)))
    print("p1 = " + str((len(seq0) - sum(prediction_res_1)) / len(seq0)))


def g0(a: float, n: int, N: int, r0: float):
    return a * np.power((1 - (n / N)), r0)


def g1(b: float, n: int, N: int, r1: float):
    return b * np.power((1 - (n / N)), r1)


def task5_theoretical_n(p0, p1, r0, r1, m0, m1, d0, d1, N):
    a = np.log((1 - p1) / p0)
    b = np.log(p1 / (1 - p0))
    E_Z_Omega_1, E_Z_Omega_0 = E_Z_Omega_1_0(m0, m1, d0)
    E_n_Omega_0 = b / (E_Z_Omega_0 + r0 * (b / N))
    E_n_Omega_1 = a / (E_Z_Omega_1 + r1 * (a / N))
    return E_n_Omega_0, E_n_Omega_1


def task5_theoretical_p(p0, p1, r0, r1, m0, m1, d0, d1, N):
    a = np.log((1 - p1) / p0)
    b = np.log(p1 / (1 - p0))
    E_Z_Omega_1, E_Z_Omega_0 = E_Z_Omega_1_0(m0, m1, d0)

    p0 = (np.exp(-a) * (1 + ((r1 * np.power(a, 2)) / (N * E_Z_Omega_1 + (r1 * a)))))
    p1 = (np.exp(b) * (1 - ((r0 * np.power(b, 2)) / (N * E_Z_Omega_0 + (r0 * b)))))
    return p0, p1


def task5_valdo_predict(p0, p1, m0, m1, d0, d1, seq, r0, r1, N):
    a = np.log((1 - p1) / p0)
    b = np.log(p1 / (1 - p0))
    a_list = []
    b_list = []
    l_list = []
    A = g0(a, 0, N, r0)
    B = g1(b, 0, N, r1)
    a_list.append(A)
    b_list.append(B)
    list_len = 1
    while A!= B:
        A = g0(a, list_len, N, r0)
        B = g1(b, list_len, N, r1)
        a_list.append(A)
        b_list.append(B)
        list_len += 1
    l = LLR(seq[0], m0, m1, d0, d1)
    l_list.append(l)
    for i in range(1, seq.size):
        if l >= a_list[i]:
            return 1, i, l_list, a_list, b_list
        if l <= b_list[i]:
            return 0, i, l_list, a_list, b_list
        l += LLR(seq[i], m0, m1, d0, d1)
        # A = g0(a, i, N, r0)
        # B = g1(b, i, N, r1)
        l_list.append(l)
        # a_list.append(A)
        # b_list.append(B)
    return np.inf, seq.size, l_list, a_list, b_list


def task5(seq0, seq1, p0, p1, m0, m1, d0, d1, r0, r1, N):
    a_list_0 = []
    b_list_0 = []
    prediction_0 = []
    count_iteration_0 = []
    l_list_0 = []
    for i in range(seq0.shape[0]):
        res = task5_valdo_predict(p0, p1, m0, m1, d0, d1, np.array(seq0[i]), r0, r1, N)
        prediction_0.append(res[0])
        count_iteration_0.append(res[1])
        l_list_0.append(res[2])
        a_list_0.append(res[3])
        b_list_0.append(res[4])
    a_list_1 = []
    b_list_1 = []
    prediction_1 = []
    count_iteration_1 = []
    l_list_1 = []
    for i in range(seq1.shape[0]):
        res = task5_valdo_predict(p0, p1, m0, m1, d0, d1, np.array(seq1[i]), r0, r1, N)
        prediction_1.append(res[0])
        count_iteration_1.append(res[1])
        l_list_1.append(res[2])
        a_list_1.append(res[3])
        b_list_1.append(res[4])
    num_for_polot = 7

    plt.scatter(np.arange(0, count_iteration_0[num_for_polot]), y=l_list_0[num_for_polot], marker="x", c="blue")
    plt.plot(np.arange(0, len(a_list_0[num_for_polot])), a_list_0[num_for_polot], marker="o", linestyle='-',
             c="red")
    plt.plot(np.arange(0, len(b_list_0[num_for_polot])), b_list_0[num_for_polot], marker="o", linestyle='-',
             c="red")
    plt.show()
    plt.scatter(np.arange(0, count_iteration_1[num_for_polot]), l_list_1[num_for_polot], marker="x", c="blue")
    plt.plot(np.arange(0, len(a_list_1[num_for_polot])), a_list_1[num_for_polot], marker="o", linestyle='-',
             c="red")
    plt.plot(np.arange(0, len(b_list_1[num_for_polot])), b_list_1[num_for_polot], marker="o", linestyle='-',
             c="red")
    plt.show()
    E_n_Omega0, E_n_Omega1 = task5_theoretical_n(p0, p1, r0, r1, m0, m1, d0, d1, N)
    theor_p0, theor_p1 = task5_theoretical_p(p0, p1, r0, r1, m0, m1, d0, d1, N)
    print("Theoretical")
    print("E{n|Ω_0} = " + str(E_n_Omega0))
    print("E{n|Ω_1} = " + str(E_n_Omega1))
    print("p0 = " + str(theor_p0))
    print("p1 = " + str(theor_p1))
    print("Practical")
    print("E{n|Ω_0} = " + str(sum(count_iteration_0) / len(count_iteration_0)))
    print("E{n|Ω_1} = " + str(sum(count_iteration_1) / len(count_iteration_1)))
    print("p0 = " + str(sum(prediction_0) / len(seq0)))
    print("p1 = " + str((len(seq0) - sum(prediction_1)) / len(seq0)))
    return


def main():
    np.random.seed(111)
    m = 50
    k = 40
    m0 = -4
    m1 = 4
    d0 = d1 = 64
    r = 0.98
    p0 = p1 = 0.3
    N = 5
    r0 = r1 = 0.25
    seq0, seq1 = task_1(m, k, m0, m1, d0, d1, r)
    seq0, seq1 = task_2(seq0, seq1, m, k)
    task_4(seq0, seq1, m0, m1, d0, d1, p0, p1, r)
    print("task_5")
    task5(seq0, seq1, p0, p1, m0, m1, d0, d1, r0, r1, N)


if __name__ == "__main__":
    main()
