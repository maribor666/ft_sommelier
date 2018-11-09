from pprint import pprint
import csv
# import matplotlib.pyplot as plt
import random


data_path = "./resources/winequality-red.csv"

# для v.2 брати x = density y= citric acidity
def main():
    reader = csv.reader(open(data_path, mode='r'), delimiter=";")
    data = [el for el in reader]
    # for el in data:
    #     print(el)
    # plot_scatter_matrix(data, 7, 4)
    part2(data)


def ft_dot(a, b):
    if len(a) != len(b):
        raise ValueError
    return sum([ai * bi for ai, bi in zip(a, b)])

def part2(data):
    X = []
    y= []
    for line in data[1:]:
        if float(line[-1]) >= 8 or float(line[-1]) <= 3:
            if float(line[-1]) >= 8:
                mark = 1
            if float(line[-1]) <= 3:
                mark = 0
            y.append(mark)
            x = [float(line[10]), float(line[8])]
            X.append(x)
    print(X)
    print(y)
    perc = Perceptron(lr=0.0005)
    perc.train(X, y)


class Perceptron:

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
                for w in self.W:
                    wnew = w + self.lr * (yi - self._heaviside(res))
                    Wnew.append(wnew)
                self.W = Wnew
                # print('Wnew', Wnew, self.bias)
        return errors_num


    def train(self, X, y, epoches=0):
        self.W = [random.uniform(-1, 1) for i in range(len(X[0]))]
        self.bias = random.uniform(-1, 1)
        print(self.W, self.bias)
        epoch_num = 0
        while True:
            epoch_erro_num = self._train_epoch(X, y)
            epoch_num += 1
            print(epoch_num, epoch_erro_num, self.W, self.bias)
            if epoch_erro_num == 0:
                break





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
