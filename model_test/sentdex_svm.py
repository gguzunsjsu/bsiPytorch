import numpy as np
import torch
import bsi_ops
import time

class SVM:
    def __init__(self):
        self.store = 0

    def fit(self, data, dot_function=torch.dot):
        # data is sent as {-1: [[1st entry], [2nd entry], ...], 1: [list of features]}
        # self.data = data

        opt_dict = {}
        transforms = torch.tensor([[1,1], [-1,-1], [-1,1], [1,-1]])

        all_data = []
        for yi in data:
            for featureset in data[yi]:
                for feature in featureset:
                    all_data.append(feature)

        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)
        del all_data

        # print(self.max_feature_value)

        step_sizes = [self.max_feature_value * 0.1,
                      self.max_feature_value * 0.01,
                      self.max_feature_value * 0.001]

        b_range_multiple = 5
        b_multiple = 5
        latest_optimum = self.max_feature_value * 10

        for step in step_sizes:
            w = torch.tensor([latest_optimum, latest_optimum], dtype=torch.float32)
            optimized = False

            while not optimized:
                for b in np.arange(-1 * (self.max_feature_value * b_range_multiple),
                               self.max_feature_value * b_range_multiple,
                               step * b_multiple):
                    for transformation in transforms:
                        w_t = w * transformation
                        found_option = True
                        for yi in data:
                            for xi in data[yi]:
                                self.store += 1
                                if not yi * (dot_function(w_t, xi) + b) >= 1:
                                    found_option = False
                                    break

                        if found_option:
                            opt_dict[np.linalg.norm(w_t)] = [w_t, b]
                            # print('Found an option,', opt_dict)

                if w[0] < 0:
                    optimized = True
                    print('Optimized a step')
                else:
                    w = w - step

            print(f'At step: {step} number of optimal choices: {len(opt_dict)}, number of dot_products done: {self.store}')
            norms = sorted(list(opt_dict.keys()))
            opt_choice = opt_dict[norms[0]]

            self.w = opt_choice[0]
            self.b = opt_choice[1]

            latest_optimum = opt_choice[0][0] + step*2

    def predict(self, testData, dot_function=torch.dot):
        res = [0] * len(testData)
        for i, features in enumerate(testData):
            classification = np.sign(dot_function(features, self.w) + self.b)
            res[i] = classification
        return res
    
def getData():
    trainData = {
        -1: torch.tensor(
            [[1, 7], [2, 8], [3, 8]],
            dtype=torch.float32
        ),
        1: torch.tensor(
            [[5, 1], [6, -1], [7, 3]],
            dtype=torch.float32
        )
    }

    testData = [
        [0, 10],
        [1, 3],
        [3, 4],
        [3, 5],
        [5, 5],
        [5, 6],
        [6, -5],
        [5, 8],
    ]
    testData = torch.tensor(testData, dtype=torch.float32)
    testResults = None

    return trainData, testData, testResults

def run():
    trainData, testData, testResults = getData()
    dot_product_functions = [torch.dot, np.dot, bsi_ops.dot_product]
    dot_product_function_names = ['torch.dot', 'np.dot', 'bsi_ops.dot_product']

    for dot_product_function, function_name in zip(dot_product_functions,dot_product_function_names):
        svm = SVM()
        fit_start = time.time()
        svm.fit(trainData, dot_function=dot_product_function)
        fit_end = time.time()
        predictions = svm.predict(testData, dot_function=dot_product_function)
        pred_end = time.time()
        print(function_name, 'Training Time:', fit_end - fit_start, 'Testing Time:', pred_end - fit_end)
        print(predictions)


    
    

if __name__ == '__main__':
    run()

