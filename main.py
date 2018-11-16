from pprint import pprint
import numpy as np
import csv
import matplotlib.pyplot as plt
import random
import pandas as pd

data_path = "./resources/winequality-red.csv"

# для v.2 брати x = density - #7 y= citric acidity # 2
def main():
    reader = csv.reader(open(data_path, mode='r'), delimiter=";")
    data = [el for el in reader]
    pd_dataframe = pd.read_csv(data_path, sep=';')
    df1 = pd_dataframe[['density', 'citric acid']]
    # print(df1)
    # plot_scatter_matrix(data, 6, 5)
    # part2(data)
    part3(data)
    # part4(pd_dataframe, data)

# part 4 start
def part4(data, csv_data):
    train_set, valid_set = holdout_part(data)
    X_valid, y_valid = prepare_values(valid_set, 8, 3)
    X_train, y_train = prepare_values(train_set, 8, 3)
    adaline = Adaline(lr=0.005)
    performance = adaline.train(X_train, y_train, mod='online', epoches=-1)
    predict_values = adaline.predict(X_valid)
    clas_errors = 0
    for y_pred, y_true in zip(predict_values, y_valid):
        if y_pred != y_true:
            clas_errors += 1
    selected_data = []
    for line in csv_data[1:]:
        selected_data.append([float(line[7]), float(line[2]), float(line[-1])])
    selected_data = normalize_data(selected_data)
    plot_performance(performance, selected_data, 8, 3)
    # errors_avg = adaline_cros_valid(data, 8, 3, folds=10)
    # print('average error number - ', errors_avg)

def adaline_cros_valid(data, good_tresh, bad_tresh, folds=5):
    folds = 5
    error_nums = []
    for train_try in range(1, folds + 1):
        train_set, valid_set = k_fold_part(data, valid_fold_num=train_try)
        X_valid, y_valid = prepare_values(valid_set, 7, 4)
        X_train, y_train = prepare_values(train_set, 7, 4)
        adaline = Adaline(lr=0.005)
        adaline.train(X_train, y_train, mod='online', epoches=500)
        predict_values = adaline.predict(X_valid)
        clas_errors = 0
        for y_pred, y_true in zip(predict_values, y_valid):
            if y_pred != y_true:
                clas_errors += 1
        clas_errors = 1 - (clas_errors / len(y_train))
        error_nums.append(clas_errors)
    return sum(error_nums) / len(error_nums)


def prepare_values(data_set, good_trash, bad_trash):
    norm_columns = []
    for column in list(data_set)[:-1]:
        norm_column = right_normilize(data_set[[column]].values)
        norm_columns.append(norm_column)
    X = []
    y = []
    for x1, x2, q in zip(norm_columns[0], norm_columns[1], data_set[['quality']].values):
        if q >= good_trash:
            y.append(1)
            X.append([x1[0], x2[0]])
        if q <= bad_trash:
            y.append(0)
            X.append([x1[0], x2[0]])
    return X, y


def right_normilize(column, mod='mean'):
    if mod == 'mean':
        min_v = min(column)
        max_v = max(column)
        avg_v = sum(column) / len(column)
        return [(x - avg_v) / (max_v - min_v) for x in column]
    if mod == 'minmax':
        min_v = min(column)
        max_v = max(column)
        return [(x - min_v) / (max_v - min_v) for x in column]


def k_fold_part(df, parts_num=5, valid_fold_num=1, shuffle=False):
    if parts_num <= 0 or valid_fold_num <= 0 or valid_fold_num > parts_num:
        raise ValueError('Doesnt correct parts num or valid fold num.')
    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)
    res = df[['density', 'citric acid', 'quality']]
    rows_num = len(res)
    slice_len = rows_num // parts_num
    start = slice_len * (valid_fold_num - 1)
    end = start + slice_len
    valid_df = res.loc[start:end-1, :]
    before_start = res.loc[:start-1, :]
    after_end = res.loc[end:, :]
    train_df = before_start.append(after_end)
    return train_df, valid_df


def holdout_part(df, parts_num=5):
    if parts_num <= 0:
        raise ValueError('Doesnt correct parts num.')
    res = df[['density', 'citric acid', 'quality']]
    rows_num = len(res)
    slice_len = rows_num // parts_num
    train_df = res.loc[:(rows_num - slice_len - 1)]
    valid_df = res[(rows_num - slice_len):]
    return train_df, valid_df

# part 4 end

# part3 start


def part3(data):
    X = []
    y = []
    for line in data[1:]:
        if float(line[-1]) >= 8 or float(line[-1]) <= 3:
            if float(line[-1]) >= 7:
                mark = 1
            if float(line[-1]) <= 4:
                mark = 0
            y.append(mark)
            X.append([float(line[7]), float(line[2])])
    selected_data = []
    for line in data[1:]:
        selected_data.append([float(line[7]), float(line[2]), float(line[-1])])
    normalized_data = normalize_data(selected_data, mod='mean')
    X_min_max = []
    y = []
    for line in normalized_data:
        if float(line[-1]) >= 8 or float(line[-1]) <= 3:
            if float(line[-1]) >= 8:
                mark = 1
            if float(line[-1]) <= 3:
                mark = 0
            y.append(mark)
            X_min_max.append([float(line[0]), float(line[1])])

    adaline = Adaline(lr=0.003)
    performance = adaline.train(X_min_max, y, mod='batch', epoches=500)
    plot_performance(performance, normalized_data, 7, 4)


class Adaline:

    def __init__(self, lr=0.01):
        self.lr = lr
        self.W = []
        self.performance = []
        self.errors_batch = []
        self.bias = None

    def _sigmoid(self, arg):
        return 1 / (1 + np.exp(-arg))

    def _activ_func(self, arg):
        return 1 if self._sigmoid(arg) >= 0.5 else 0

    def _train_epoch_online(self, X, y):
        errors_num = 0
        for xi, yi in zip(X, y):
            res = ft_dot(xi, self.W) + self.bias
            if self._activ_func(res) != yi:
                errors_num += 1
                self.bias = self.bias + self.lr * (yi - self._activ_func(res))
                Wnew = []
                for xi_j, w in zip(xi, self.W):
                    wnew = w + self.lr * (yi - self._activ_func(res)) * xi_j
                    Wnew.append(wnew)
                self.W = Wnew
        return errors_num

    def _train_epoch_batch(self, X, y):
        errors_num = 0
        erros_batch = []
        for xi, yi in zip(X, y):
            res = ft_dot(xi, self.W) + self.bias
            if self._activ_func(res) != yi:
                errors_num += 1
                erros_batch.append([xi, yi])
        self.errors_batch.append(erros_batch)

        for xi, yi in erros_batch:
            res = ft_dot(xi, self.W) + self.bias
            self.bias = self.bias + self.lr * (yi - self._activ_func(res))
            Wnew = []
            for xi_j, w in zip(xi, self.W):
                wnew = w + self.lr * (yi - self._activ_func(res)) * xi_j
                Wnew.append(wnew)
            self.W = Wnew
        return errors_num

    def train(self, X, y, epoches=-1, mod='online'):
        self.W = [random.uniform(-1, 1) for i in range(len(X[0]))]
        self.bias = random.uniform(-1, 1)
        epoch_num = 0
        if mod == 'online':
            while True:
                epoch_erro_num = self._train_epoch_online(X, y)
                epoch_num += 1
                self.performance.append((epoch_num, epoch_erro_num, self.W, self.bias))
                pprint((epoch_num, epoch_erro_num, self.W, self.bias))
                if epoches == epoch_num and epoches > 0:
                    break
                if epoch_erro_num == 0:
                    break
            return self.performance
        elif mod == 'batch':
            while True:
                epoch_erro_num = self._train_epoch_batch(X, y)
                epoch_num += 1
                self.performance.append((epoch_num, epoch_erro_num, self.W, self.bias))
                pprint((epoch_num, epoch_erro_num, self.W, self.bias))
                if epoches == epoch_num and epoches > 0:
                    break
                if epoch_erro_num == 0:
                    break
            return self.performance
        else:
            raise ValueError('Mod isnt correct')

    def predict(self, X):
        predicted_values = []
        for xi in X:
            res = ft_dot(xi, self.W) + self.bias
            predicted_values.append(self._activ_func(res))
        return predicted_values

# part3 end
# part2 start


def ft_dot(a, b):
    if len(a) != len(b):
        raise ValueError
    return sum([ai * bi for ai, bi in zip(a, b)])


def part2(data):
    X = []
    y = []
    for line in data[1:]:
        if float(line[-1]) >= 8 or float(line[-1]) <= 3:
            if float(line[-1]) >= 7:
                mark = 1
            if float(line[-1]) <= 4:
                mark = 0
            y.append(mark)
            X.append([float(line[7]), float(line[2])])
    perc = Perceptron(lr=0.005)
    # performance = perc.train(X, y, 0)
    selected_data = []
    for line in data[1:]:
        selected_data.append([float(line[7]), float(line[2]), float(line[-1])])
    # plot_performance(performance, selected_data, 8, 3)


    normalized_data = normalize_data(selected_data, mod='minmax')
    X_min_max = []
    y = []
    for line in normalized_data:
        if float(line[-1]) >= 8 or float(line[-1]) <= 3:
            if float(line[-1]) >= 8:
                mark = 1
            if float(line[-1]) <= 3:
                mark = 0
            y.append(mark)
            X_min_max.append([float(line[0]), float(line[1])])
    perc = Perceptron(lr=0.001)
    performance = perc.train(X_min_max, y, 0)
    pprint(performance)
    plot_performance(performance, normalized_data, 7, 4)


def normalize_data(selected_data, mod='mean'):
    if mod == 'minmax':
        first_column = [el[0] for el in selected_data]
        first_max = max(first_column)
        first_min = min(first_column)
        norm_first = [(x - first_min) / (first_max - first_min) for x in first_column]
        second_column = [el[1] for el in selected_data]
        second_max = max(second_column)
        second_min = min(second_column)
        norm_second = [(x - second_min) / (second_max - second_min) for x in second_column]
        norm_data = []
        quals = [el[-1] for el in selected_data]
        for first, second, qual in zip(norm_first, norm_second, quals):
            norm_data.append([first, second, qual])
        return norm_data
    elif mod == 'mean':
        first_column = [el[0] for el in selected_data]
        first_max = max(first_column)
        first_min = min(first_column)
        first_avg = sum(first_column) / len(first_column)
        norm_first = [(x - first_avg) / (first_max - first_min) for x in first_column]
        second_column = [el[1] for el in selected_data]
        second_max = max(second_column)
        second_min = min(second_column)
        second_avg = sum(second_column) / len(second_column)
        norm_second = [(x - second_avg) / (second_max - second_min) for x in second_column]
        norm_data = []
        quals = [el[-1] for el in selected_data]
        for first, second, qual in zip(norm_first, norm_second, quals):
            norm_data.append([first, second, qual])
        return norm_data


def plot_performance(performance, data, good_thresh, bad_thresh, epoch=-1, save_plot=False):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].set_xlabel('epoch')
    axes[0].set_ylabel('classification errors')
    axes[0].set_title('Errors as a function of epoch')
    if epoch > len(performance):
        raise Exception('too large epoch num passed')
    if epoch > 0:
        errors = [el[1] for el, feature in zip(performance, range(epoch))]
        epoches = [el[0] for el, feature in zip(performance, range(epoch))]
    else:
        errors = [el[1] for el in performance]
        epoches = [el[0] for el in performance]
    axes[0].plot(epoches, errors)
    if epoch > 0:
        W = performance[epoch][2]
        bias = performance[epoch][3]
    else:
        W = performance[-1][2]
        bias = performance[-1][3]
    w = -(W[0] / W[1])
    b = -bias/W[1]
    x = [x for x in np.arange(-15, 15, .01)]
    y = [(w * xi + b) for xi in x]
    axes[1].plot(x, y, 'b--', linewidth='1', label='Decision boundary')
    axes[1].set_xlabel('density')
    axes[1].set_ylabel('citric acid')
    if epoch > 0:
        epoch_last = epoch
    else:
        epoch_last = performance[-1][0]
    axes[1].set_title('Decision boundary on epoch: ' + str(epoch_last))
    good_x, good_y, bad_x, bad_y = [], [], [], []
    for line in data:
        if line[-1] >= good_thresh or line[-1] <= bad_thresh:
            if line[-1] >= good_thresh:
                good_x.append(line[0])
                good_y.append(line[1])
            if line[-1] <= bad_thresh:
                bad_x.append(line[0])
                bad_y.append(line[1])
    dots1 = axes[1].plot(good_x, good_y, 'go', label='good wines(>=' + str(good_thresh) + ' score)')
    dots2 = axes[1].plot(bad_x, bad_y, 'ro', label='bad wines(<=' + str(bad_thresh) + ' score)')
    good_x.extend(bad_x)
    good_y.extend(bad_y)
    min_x_axes = min(good_x) - 0.01
    max_x_axes = max(good_x) + 0.01
    min_y_axes = min(good_y) - 0.01
    max_y_axes = max(good_y) + 0.01
    axes[1].set_xlim([min_x_axes, max_x_axes])
    axes[1].set_ylim([min_y_axes, max_y_axes])
    dots1[0].set_markersize(1.5)
    dots2[0].set_markersize(1.5)
    axes[1].fill_between(x, y, min_y_axes, color='#D89797')
    axes[1].fill_between(x, y, max_y_axes, color='#B8F4C0')
    box = axes[1].get_position()
    axes[1].set_position([box.x0, box.y0, box.width * 0.8, box.height])
    axes[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()
    if save_plot:
        fig.savefig("./decision_boundary.png")


class Perceptron():

    def __init__(self, lr=0.0001):
        self.lr = lr
        self.performance = []
        self.W = []
        self.bias = None

    def _heaviside(self, arg):
        return 1 if arg >= 0 else 0

    def _train_epoch(self, X, y):
        errors_num = 0
        for xi, yi in zip(X, y):
            res = ft_dot(self.W, xi) + self.bias
            if self._heaviside(res) != yi:
                errors_num += 1
                self.bias = self.bias + self.lr * (yi - self._heaviside(res))
                Wnew = []
                for w, xi_j in zip(self.W, xi):
                    wnew = w + self.lr * (yi - self._heaviside(res)) * xi_j
                    Wnew.append(wnew)
                self.W = Wnew
        return errors_num

    def train(self, X, y, epoches=0):
        self.W = [random.uniform(-1, 1) for i in range(len(X[0]))]
        self.bias = random.uniform(-1, 1)
        epoch_num = 0
        while True:
            epoch_erro_num = self._train_epoch(X, y)
            epoch_num += 1
            self.performance.append((epoch_num, epoch_erro_num, self.W, self.bias))
            if epoches == epoch_num:
                break
            if epoch_erro_num == 0:
                break
        return self.performance
#part2 end

#part1 start
def plot_scatter_matrix(wine_data, good_treshold, bad_trashold, save_plot=False):
    count_of_props = len(wine_data[0]) - 1
    fig, axes = plt.subplots(count_of_props, count_of_props, figsize=(15, 10))
    fig.subplots_adjust(hspace=0, wspace=0)
    prop_names = wine_data[0][:-1]
    i = 0
    for ax in axes:
        ax[i].text(0.5, 0.5, prop_names[i].replace(' ', '\n'), ha='center', va='center')
        for a in ax:
            a.set_xticklabels([])
            a.set_yticklabels([])
            a.tick_params(axis='both', width=0)
        i += 1
    columns = []
    for column in range(len(wine_data[0])):
        columns.append([el[column] for el in wine_data])
    columns2 = []
    for column in columns:
        columns2.append([float(el) for el in column[1:]])
    columns = columns2
    column_height = len(columns[0])
    for i in range(count_of_props):
        for j in range(count_of_props):
            if i == j:
                continue
            good_x = []
            good_y = []
            bad_x = []
            bad_y = []
            for k in range(column_height):
                if columns[-1][k] > good_treshold:
                    good_x.append(columns[j][k])
                    good_y.append(columns[i][k])
                if columns[-1][k] < bad_trashold:
                    bad_x.append(columns[j][k])
                    bad_y.append(columns[i][k])
            dots1 = axes[i][j].plot(good_x, good_y, 'go')
            dots1[0].set_markersize(0.5)
            dots2 = axes[i][j].plot(bad_x, bad_y, 'ro')
            dots2[0].set_markersize(0.5)
    if save_plot:
        plt.savefig("./scatter_plot.png")
    plt.show()
#part1 end

if __name__ == '__main__':
    main()