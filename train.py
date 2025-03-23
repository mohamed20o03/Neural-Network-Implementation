from Value import Value
from NeuralNetworks import MLP

def main():
    X = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]
    Y = [1.0, -1.0, -1.0, 1.0]

    model = MLP(3, [4, 4, 1])
    ypred = train(model, X, Y)
    print("Final predictions:", ypred)

def train(model, X_Train, y_train):

    for k in range(100):
        # forward pass
        ypred = [model(x) for x in X_Train]
        loss = sum((yout - ygt) ** 2 for ygt, yout in zip(y_train, ypred))

        # backward pass
        for p in model.parameters:
            p.grad = 0.0
        loss.backward()

        # update
        for p in model.parameters:
            p.data += -0.1 * p.grad

        print(k, loss.data)

    return ypred

if __name__ == "__main__":
    main()
