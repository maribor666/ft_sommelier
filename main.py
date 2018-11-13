from pprint import pprint
import csv
import matplotlib.pyplot as plt
import random
import pandas as pd

data_path = "./resources/winequality-red.csv"

# для v.2 брати x = density y= citric acidity
def main():
    reader = csv.reader(open(data_path, mode='r'), delimiter=";")
    data = [el for el in reader]
    pd_dataframe = pd.read_csv(data_path, sep=';')
    df1 = pd_dataframe[['density', 'citric acid' ]]
    # print(df1)
    # plot_scatter_matrix(data, 7, 4)
    part2(data)


def ft_dot(a, b):
    if len(a) != len(b):
        raise ValueError
    return sum([ai * bi for ai, bi in zip(a, b)])

def part2(data):
    X = []
    y = []
    for line in data[1:]:
        if float(line[-1]) >= 8 or float(line[-1]) <= 3:
            if float(line[-1]) >= 8:
                mark = 1
            if float(line[-1]) <= 3:
                mark = 0
            y.append(mark)
            X.append([float(line[7]), float(line[2])])
    min_x1 = min([el[0] for el in X])
    max_x1 = max([el[0] for el in X])
    min_x2 = min([el[1] for el in X])
    max_x2 = max([el[1] for el in X])

    X_min_max  = []
    for x1, x2 in X:
        x1_new = (x1 - min_x1)/(max_x1 - min_x1)
        x2_new = (x2 - min_x2)/(max_x2 - min_x2)
        X_min_max.append([x1_new, x2_new])
    perc = Perceptron(lr=0.005)
    performance = perc.train(X_min_max, y, 0)
    for el in performance:
        print(el)

    selected_data = []
    for line in data[1:]:
        selected_data.append([float(line[7]), float(line[2]), float(line[-1])])
    normalized_data = normalize_data(selected_data, mod='minmax')
    pprint(normalized_data)
    plot_performance(performance, normalized_data, 8, 3, epoch=0, save_plot=False)


def normalize_data(selected_data, mod='minmax'):
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



def plot_performance(performance, data, good_thresh, bad_thresh, epoch=-1, save_plot=False):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.canvas.set_window_title('Training with perceptron')
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
    x = [x for x in range(-15, 15)]
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


class Perceptron(object):

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
                for w, xi_i in zip(self.W, xi):
                    wnew = w + self.lr * (yi - self._heaviside(res)) * xi_i
                    Wnew.append(wnew)
                self.W = Wnew
        return errors_num

    def train(self, X, y, epoches=0):
        pprint(X)
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


if __name__ == '__main__':
    main()
